
# Import custom libraries and modules
from lit_review_machine import config, utils
from lit_review_machine.prompts import Prompts

# Import standard libraries
from typing import Any, Dict, Optional
import os
import pandas as pd
import unicodedata
import re
from docx import Document
import json


class Summaries:
    def __init__(
        self, 
        llm_client: Any,
        ai_model: str,
        summaries: Optional[pd.DataFrame] = None, 
        summaries_folder: str = os.path.join(config.SUMMARY_SAVE_LOCATION, "parquet"),
        summary_string: Optional[str] = None, 
        ai_peer_review: Optional[Dict] = None,
        output_save_location: str = os.path.join(config.SUMMARY_SAVE_LOCATION, "results")
        ):
        """
        Wrapper for a DataFrame of summaries and tools for AI-assisted peer review.
        
        Args:
            summaries: DataFrame of either the summaries or the summaries synthesized into themes
            llm_client: LLM client for interacting with the AI model.
            ai_model: Name of the deep research model to use.
            summary_string: Optional; pre-computed concatenated summary string.
            ai_peer_review: Optional; stores the AI peer review output as a dictionary.
            output_save_location: Directory to save Word documents.
        """
        
        
        # If summaries is not provided, try and load from parquet
        if summaries is None:
            self.summaries = self.from_parquet(summaries_folder=summaries_folder, llm_client=llm_client, ai_model=ai_model)


        self.summaries: pd.DataFrame = summaries
        self.llm_client = llm_client
        self.ai_model = ai_model
        self.summary_string: Optional[str] = summary_string
        self.ai_peer_review: Optional[Dict] = ai_peer_review
        self.summaries_folder: str = summaries_folder
        self.output_save_location: str = output_save_location

        
    @classmethod
    def from_parquet(cls, summaries_folder: str, llm_client: Any, ai_model: str) -> "Summaries":
        """
        Load summaries from a parquet file containing a DataFrame.

        Args:
            filepath: Path to the parquet file.

        Returns:
            Summaries instance with loaded DataFrame.
        """

        files = os.listdir(cls.output_save_location)
        summaries_file = [file for file in files if file in ["summaries.parquet", "clean_summaries.parquet"]]
        if len(summaries_file) == 0:
            raise FileNotFoundError(f"No 'summaries.parquet' or 'clean_summaries.parquet' file found in {cls.output_save_location}, cannot load Summaries.")
        elif len(summaries_file) > 1:
            raise ValueError(f"Multiple summary files found in {cls.output_save_location}. Expected only one of 'summaries.parquet' or 'clean_summaries.parquet'. Either delete on file and retry, or load summary manually and pass to the constructor.")

        summaries = pd.read_parquet(os.path.join(summaries_folder, summaries_file[0]))

        return cls(summaries = summaries, 
                   llm_client = llm_client, 
                   ai_model = ai_model)

    def get_summary_string(self, output_result: bool = True) -> Optional[str]:
        """
        Concatenate cluster summaries into a single string per research question,
        ordered by question_id and cluster.

        Args:
            output_result: If True, return the concatenated string; else only sets self.summary_string.

        Returns:
            Concatenated summary string if output_result is True; else None.
        """
        # Ensure DataFrame is sorted by question_id and cluster for stable ordering
        self.summaries = self.summaries.sort_values(by=["question_id", "cluster"])

        output_string = ""
        for qid in self.summaries["question_id"].unique():
            qtext = self.summaries.loc[self.summaries["question_id"] == qid, "question_text"].iloc[0]
            question_df = self.summaries[self.summaries["question_id"] == qid]

            question_string = (
                f"Research question id: {qid}\n"
                f"Research question text: {qtext}\n"
                "Review:\n"
                f"{'\n\n'.join(question_df['cluster_summary'].tolist())}\n\n"
            )
            output_string += question_string

        self.summary_string = output_string

        if output_result:
            return output_string
   
    def gen_executive_summary(self, token_length: int = 600) -> Optional[str]:
        if not hasattr(self, "summaries"):
            raise ValueError("summaries attribute not found.")

        df = self.summaries.copy()
        df = df.reset_index(drop=False).rename(columns={"index": "_row"})  # preserve original order
        themed = {"label", "contents"}.issubset(df.columns)

        parts: list[str] = []
        for qtext, qdf in df.groupby("question_text", sort=False):
            parts.append(f"Question: {qtext}\n")
            qdf = qdf.sort_values("_row")
            if themed:
                for _, r in qdf.iterrows():
                    label = (r.get("label") or "").strip()
                    content = (r.get("contents") or "").strip()
                    if not content:
                        continue
                    parts.append(
                        f"Theme: {label}\n"
                        f"{content}\n"
                        "--- END THEME ---\n"
                    )
            else:
                for _, r in qdf.iterrows():
                    summ = (r.get("summary") or "").strip()
                    if not summ:
                        continue
                    parts.append(f"{summ}\n")
            parts.append("=== END QUESTION ===\n")

        if not parts:
            return None

        all_text = "\n".join(parts).strip()

        sys_prompt = Prompts().exec_summary(token_length=token_length)
        resp = utils.call_chat_completion(
            sys_prompt=sys_prompt,
            user_prompt=all_text,
            llm_client=self.llm_client,
            ai_model=self.ai_model,
            return_json=True,
            fall_back={"executive_summary": "", "title": ""},
        )

        summary = (resp.get("executive_summary") or "").strip()
        title = (resp.get("title") or "").strip()
        self.exec_summary = summary or None
        self.title = title or None
        return self.title, self.exec_summary


    def gen_question_summaries(self) -> Optional[pd.DataFrame]:
        # Preconditions
        if not hasattr(self, "summaries"):
            return None
        df = self.summaries
        if "question_text" not in df.columns:
            raise ValueError("summaries must contain 'question_text'")

        # Decide input mode
        has_themes = {"label", "contents"}.issubset(df.columns)
        has_raw    = "summary" in df.columns
        if not (has_themes or has_raw):
            raise ValueError("summaries must have either ['label','contents'] or ['summary']")

        out_rows = []

        for qtext, qdf in df.groupby("question_text", sort=False):
            # Build payload for this question
            parts = [f"Question: {qtext}", ""]
            if has_themes:
                for _, r in qdf.iterrows():
                    parts.append(f"Theme: {r['label']}\n{r['contents']}\n")
            else:
                for _, r in qdf.iterrows():
                    parts.append(f"{r['summary']}\n")
            payload = "\n".join(parts).strip()

            # Call LLM
            sys_prompt = Prompts().question_summaries()
            fall_back = {"summary": ""}

            resp = utils.call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=payload,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                return_json=True,
                fall_back=fall_back,
            )

            out_rows.append({
                "question_text": qtext,
                "question_summary": (resp.get("summary") or "").strip(),
            })

        if not out_rows:
            return None

        qsum = (
            pd.DataFrame(out_rows)
            .drop_duplicates(subset=["question_text"], keep="last")
        )

        # One summary per question_text
        self.summaries = self.summaries.merge(qsum, on="question_text", how="left", validate="m:1")
        return self.summaries


    def summary_to_doc(self, paper_title: str = None, summary_filename: str = None) -> str:
        
        def _sanitize(text: str) -> str:
            control_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
            if text is None:
                return ""
            # ensure str
            s = str(text)
            # normalize unicode
            s = unicodedata.normalize("NFC", s)
            # drop XML-illegal control chars (keep \t, \n, \r)
            s = control_chars.sub('', s)
            return s
        
        if summary_filename is None:
            summary_filename = "literature_review.docx"

        if paper_title is None and hasattr(self, "title"):
            paper_title = self.title
        elif paper_title is None:
            raise ValueError("Paper_title must be provided. Either pass it in summary_to_doc or call get_executive_summary() which will provide a title.")

        doc = Document()
        doc.add_heading(_sanitize(paper_title), level=0)

        # Executive summary
        if getattr(self, "exec_summary", None):
            doc.add_heading("Executive summary", level=1)
            for para in str(self.exec_summary).splitlines():
                p = _sanitize(para)
                if p.strip():
                    doc.add_paragraph(p)
            doc.add_page_break()
        else: 
            raise ValueError("exec_summary attribute not found. Please run gen_executive_summary() before exporting to doc.")

        df = self.summaries.copy().reset_index(drop=False).rename(columns={"index": "_row"})
        themed = {"label", "contents"}.issubset(df.columns)

        if themed:
            for question_text, qdf in df.groupby("question_text", sort=False):
                doc.add_heading(_sanitize(question_text), level=1)
                if "question_summary" in qdf.columns:
                    qs = (qdf["question_summary"].dropna().astype(str).iloc[0]
                        if not qdf["question_summary"].dropna().empty else "")
                    if qs.strip():
                        doc.add_paragraph(_sanitize(qs))
                qdf = qdf.sort_values("_row")
                for _, r in qdf.iterrows():
                    label = _sanitize(r.get("label", ""))
                    contents = _sanitize(r.get("contents", ""))
                    if label:
                        doc.add_heading(f"Theme: {label}", level=2)
                    if contents:
                        for para in contents.splitlines():
                            p = _sanitize(para)
                            if p.strip():
                                doc.add_paragraph(p)
                doc.add_page_break()
        else:
            for question_text, qdf in df.groupby("question_text", sort=False):
                doc.add_heading(_sanitize(question_text), level=1)
                if "question_summary" in qdf.columns:
                    qs = (qdf["question_summary"].dropna().astype(str).iloc[0]
                        if not qdf["question_summary"].dropna().empty else "")
                    if qs.strip():
                        doc.add_paragraph(_sanitize(qs))
                qdf = qdf.sort_values("_row")
                for _, r in qdf.iterrows():
                    summ = _sanitize(r.get("summary", ""))
                    if summ:
                        for para in summ.splitlines():
                            p = _sanitize(para)
                            if p.strip():
                                doc.add_paragraph(p)
                doc.add_page_break()

        
        # Save
        save_dir = self.output_save_location
        os.makedirs(save_dir, exist_ok=True)
        
        out_path = os.path.join(save_dir, summary_filename)
        doc.save(out_path)
        
        # Mark that doc has been generated - as flag for running the AI peer review (neccesary as all the steps for running the dog generation are needed for peer reviewn).
        self.doc = True

        return (
            f'Word doc of the literature review generated and save here: {out_path}'
        )
    
    def get_ai_peer_review(self, 
                           save_directory: str = None,
                           save_filename: str = "ai_peer_review.parquet",
                           output_length: int = 5000, 
                           max_tokens: int = 10000) -> pd.DataFrame:
        """
        Request an AI peer review of the concatenated summaries.

        Args:
            output_length: Suggested maximum word count for the review.
            max_tokens: Hard limit on token usage for the AI model.

        Returns:
            AI review as a dataframe. If the review exceeds token budget, 'error' key may appear.
        """        
        # Function specific utilities ---------------------
        # Get the RQ and summaries excluding the current RQ
        def get_rq_summaries(self, current_rq_id) -> str:
            output_parts = []
            for rq_id, rq_df in self.summaries.groupby("question_id", sort=False):
                if rq_id == current_rq_id:
                    continue
                else:
                    output_parts.append(f"RESEARCH QUESTION:\n{rq_id}\n")
                    question_summary = rq_df["question_summary"].iloc[0]
                    output_parts.append(f"SUMMARY:\n{question_summary}\n")
            return "\n".join(output_parts)
        
        # Get the RQ and its associated content
        def get_rq_content(self, current_rq_id) -> str:
            current_rq_df = self.summaries[self.summaries["question_id"] == current_rq_id]
            current_rq_text = current_rq_df["question_text"].iloc[0]
            output_parts = []
            output_parts.append(f"RESEARCH QUESTION:\n{current_rq_text}\n")
            themed = {"label", "contents"}.issubset(self.summaries.columns)
            if themed:
                for _, row in current_rq_df.iterrows():
                    label = row["label"].strip()
                    content = row["contents"].strip()
                    output_parts.append(
                        f"Theme: {label}\n"
                        f"Summary: {content}\n"
                    )

                return "\n".join(output_parts)
            
            else:
                for _, row in current_rq_df.iterrows():
                    summary = row["summary"].strip()
                    output_parts.append(f"Summary: {summary}\n")
                return "\n".join(output_parts)
        
        # END UTILS -------------------------------

        # Check that executive summary has been generated
        if getattr(self, "exec_summary", None) is None:
            raise ValueError("Please run gen_executive_summary() before requesting an AI peer review.")
        
        # Check that the peer review has not already been generated
        if save_directory is None:
            save_directory = self.summaries_folder

        if os.path.exists(os.path.join(save_directory, save_filename)):
            recover = None
            while recover not in ["r", "n"]:
                recover = input("AI peer review file already exists. Loading existing file. (r)eload or create (n)ew? ")
            if recover == "r":
                self.ai_peer_review = pd.read_parquet(os.path.join(save_directory, save_filename))
                return self.ai_peer_review
            else:
                print("Generating new AI peer review and overwriting existing file.")

        # Populate the list that will form the prompt for the LLM
        output_list = []
        for rq_id, rq_df in self.summaries.groupby("question_id", sort=False):
            output_parts = []
            output_parts.append("INFORMATION FOR CONTEXT\n")
            output_parts.append("EXECUTIVE SUMMARY:\n")
            output_parts.append(self.exec_summary + "\n")
            output_parts.append("OTHER RESEARCH QUESTIONS AND SUMMARIES:\n")
            output_parts.append(get_rq_summaries(self, current_rq_id=rq_id))
            output_parts.append("CURRENT RESEARCH QUESTION AND CONTENT:\n")
            output_parts.append(get_rq_content(self, current_rq_id=rq_id))

            output_string = "\n".join(output_parts)
            
            # Call the reasoning model
            resp = utils.call_reasoning_model(
                prompt=Prompts().ai_peer_review(
                    lit_review=output_string,
                    output_length=output_length,
                    max_tokens=max_tokens,
                    themed={"label", "contents"}.issubset(self.summaries.columns)
                ),
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                timeout=1200
            )

            # Handle the response with a retry to clean the JSON via an LLM
            try:
                response_dict = json.loads(resp)
            except json.JSONDecodeError:
                response_retry = utils.llm_json_clean(x = resp,
                                                prompt = Prompts().peer_review_format_check(),
                                                llm_client=self.llm_client,
                                                ai_model=self.ai_model)
                response_dict = json.loads(response_retry)

            overall_comment_df = pd.DataFrame({"comment_id": "overall", 
                                               "comment": response_dict.get("overall_comment"),
                                               "severity": pd.NA, 
                                               "location": "overall"}, index=[0])
                 

            specific_comments_df = pd.DataFrame(
                response_dict.get("specific_comments", []), 
                columns=["comment_id", "comment", "severity", "location"]
            )


            output = pd.concat([overall_comment_df, specific_comments_df], ignore_index=True)
            output["research_question_id"] = rq_id
            output["research_question_text"] = rq_df["question_text"].iloc[0]
            output["resubmit"] = response_dict.get("resubmit", False)

            output_list.append(output)

        output_df = pd.concat(output_list).reset_index(drop=True)
        # Convert to string and fill missing severity with "None" so that parquet will accept it
        output_df["severity"] = output_df["severity"].apply(lambda x: x if pd.notna(x) else "None")
        # Also make all id str (handle int and "overall" fields)
        output_df["comment_id"] = output_df["comment_id"].astype(str)

        self.ai_peer_review = output_df

        os.makedirs(save_directory, exist_ok=True)
        self.ai_peer_review.to_parquet(os.path.join(save_directory, save_filename), index=False)
        
        return self.ai_peer_review
    
    def peer_review_to_doc(self):
        if self.ai_peer_review is None:
            print("No AI peer review data available. Please run get_ai_peer_review() first.")
            return None
        
        # Convert the AI peer review DataFrame to a formatted string or document
        # This is a placeholder for actual document generation logic
        doc = Document()
        for _, row in self.ai_peer_review.iterrows():
            doc.add_paragraph(f"Comment ID: {row['comment_id']}")
            if row["severity"] is not None:
                doc.add_paragraph(f"Severity: {row['severity']}")
            if row["resubmit"]:
                doc.add_paragraph("**Resubmission Required**")
            if row["location"] == "overall":
                doc.add_paragraph(f"Location: {row['research_question_text']}")
            else:
                doc.add_paragraph(f"Location: {row['research_question_text']} - {row['location']}")
            doc.add_paragraph(f"Comment: {row['comment']}")
            doc.add_paragraph("---------------------")


        # Save
        save_dir = self.output_save_location
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, "peer_review.docx")
        doc.save(out_path)

        print(f"Peer review generated and saved to {out_path}")
        