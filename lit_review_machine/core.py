
# import custom libraries

from lit_review_machine import config, utils
from lit_review_machine.state import QuestionState
from lit_review_machine.render import Summaries
from lit_review_machine.prompts import Prompts

# import standard libraries
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from copy import deepcopy
import os
from pathlib import Path
from collections import defaultdict
import json
import pymupdf
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import tiktoken
import re
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import itertools
import networkx as nx


class Ingestor:
    """
    Class to ingest PDF or HTML papers into a QuestionState object.
    Validates papers against known question_ids and populates state.full_text.

    Attributes:
        state: QuestionState object containing literature metadata.
        file_path: Directory containing PDF/HTML files to ingest.
        llm_client: Client for calling the LLM.
        ai_model: Model name to use for LLM.
        confirm_read: Optional; set to "c" to skip ingestion error confirmation.
        ingestion_errors: List of file paths that failed ingestion.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_model: str,
        state: Optional[QuestionState] = None,
        questions: Optional[List[str]] = None,
        papers: Optional[pd.DataFrame] = None,
        file_path: str = os.path.join(os.getcwd(), "data", "docs"),
        pickle_path: str = config.PICKLE_SAVE_LOCATION # For storing the pickles of LLM metadata retreival for resume
    ) -> None:
        """Initialize Ingestor and validate state/papers format."""
               
        self.state = deepcopy(
            utils.validate_format(
                state=state,
                questions=questions,
                injected_value=papers,
                state_required_cols=[
                    "question_id", "question_text", "search_string_id", "search_string",
                    "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                    "download_status", "messy_question_id", "messy_paper_id"
                ],
                injected_required_cols=["question_id", "question_text"]
            )
        )

        self.state.enforce_canonical_question_text()
        self.file_path: str = file_path
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.ingestion_errors: List[str] = []
        self.pickle_path: str = pickle_path


    @staticmethod
    def _pprint_dict(d: dict):
        """Pretty print a dictionary."""
        lines = []
        # Attach keys to a list of their values with indentation for readability and with newlines
        for key, value in d.items():
            lines.append(f"{key}\n")
            for item in value:
                lines.append(f"    {item}\n")
            lines.append("\n")
        # Convert to a string which when retured will print the dictionary in a readable format with keys and indented values
        line_str = "".join(lines)

        return(line_str)
                     
    def _list_files(self) -> List[str]:
        """
        Recursively list all PDF and HTML files in the target directory.
        Identify duplicate file names to maintain unique paper_id across the data.
        Show the user any conflicting paths with absolute path to allow them to resolve before ingestion.
        
        """
        list_of_path_obj: List[str] = []
        # First get all the files that are html and pdfs in the directory and subdirectories and store as path objects in a list
        for root, _, files in os.walk(self.file_path):
            for file in files:
                p = Path(root) / file
                if p.is_file() and p.suffix.lower() in [".pdf", ".html"]:
                     list_of_path_obj.append(p)

        
        # Then check for duplicate file names (not paths) in the list of path objects - this is important as the paper_id is derived from the file name, 
        # so if there are duplicate file names this will cause issues with linking papers to insights later in the pipeline. 
        # If there are duplicate file names, print out the conflicting file names and their absolute paths to allow the user to resolve before ingestion.
        list_of_files = [p.name for p in list_of_path_obj]
        # If there are duplicates find them and raise an error with the full path for the user
        if len(list_of_files) != len(set(list_of_files)):
            conflicting_files = defaultdict(list)
            for file_name in list_of_files:
                if list_of_files.count(file_name) > 1:
                    for f, p in zip(list_of_files, list_of_path_obj):
                        if f == file_name:
                            conflicting_files[file_name].append(p.resolve())
            raise ValueError(
                "Duplicate file names detected in ingestion directory. " 
                "Please ensure all files have unique names to avoid ingestion errors." 
                "Conflicting files and their paths are as follows:\n" 
                + self._pprint_dict(conflicting_files)
            )
        
        # Otherwise return the list of files
        return [p.absolute() for p in list_of_path_obj]

    def _ingest_pdf(self, path: str) -> List[str]:
        """Extract text from all pages of a PDF file."""
        with pymupdf.open(path) as doc:
            return [doc[i].get_text() for i in range(doc.page_count)]

    @staticmethod
    def _html_cleaner(html_content: str) -> str:
        """Clean HTML content by removing structural noise and returning plain text."""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)
        return ""

    @staticmethod
    def _html_chunker(clean_html: str, token_limit: int = 16000) -> List[str]:
        """Split HTML text into chunks if it exceeds the token limit."""
        if len(clean_html) == 0:
            return [""]
        elif len(clean_html) > token_limit:
            chunks: List[str] = []
            start = 0
            end = token_limit
            while start < len(clean_html):
                chunks.append(clean_html[start:end])
                start += token_limit
                end += token_limit
            return chunks
        else:
            return [clean_html]

    def _llm_parse_html(self, html_list: List[str], prompt: str) -> List[str]:
        """Call the LLM to extract meaningful content from HTML chunks."""
        if html_list[0] == "":
            return [""]
        output: List[str] = []
        for chunk in html_list:
            sys_prompt = prompt
            user_prompt = f"[START_TEXT] {chunk} [END_TEXT]"
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.llm_client.chat.completions.create(
                model=self.ai_model,
                messages=messages
            )
            output.append(response.choices[0].message.content)
        return output

    def _paper_ingestor(self, file_full_path: str) -> List[str]:
        """Read PDF or HTML file and return list of page texts or processed chunks."""
        if Path(file_full_path).suffix.lower() == ".pdf":
            return self._ingest_pdf(file_full_path)
        
        elif Path(file_full_path).suffix.lower() == ".html":
            with open(file_full_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            clean_html = self._html_cleaner(html_content)
            html_chunks = self._html_chunker(clean_html)
            print("File is html, sending to LLM for final parsing...")
            return self._llm_parse_html(html_chunks, prompt=Prompts().extract_main_html_content())
        else:
            return ["Unsupported file type"]

    def ingest_papers(self) -> pd.DataFrame:
        """
        Ingest all papers and populate state.full_text.
        Returns a DataFrame with columns ['paper_path', 'pages', 'paper_id', 'question_id', 'full_text'].
        """

        
        working_insights = (
            self.state.insights.copy()
            .assign(paper_id=lambda x: x["paper_id"].astype(str)) # set as string to handle NaN values so avoid merge issues on str and float (NaN)
        )

        list_of_papers_by_page: List[List[str]] = []
        ingestion_status: List[int] = [] # to track ingestion success (1) or failure (0)
        self.ingestion_errors = []

        list_of_files = self._list_files()
        if len(list_of_files) == 0:
            raise ValueError(f"No PDF or HTML files found in the specified directory: {self.file_path}. Please add files to ingest or check the directory path.")

        for count, file in enumerate(list_of_files, start=1):
            print(f"Ingesting paper {count} of {len(list_of_files)}...")
            try:
                pages = self._paper_ingestor(file)
                list_of_papers_by_page.append(pages)
                ingestion_status.append(1)
            except Exception as e:
                list_of_papers_by_page.append([str(e)])
                self.ingestion_errors.append(file)
                ingestion_status.append(0)

        # Confirm ingestion errors
        if self.ingestion_errors:
            abort_failed_ingest = None
            while abort_failed_ingest not in ["y", "n"]:
                abort_failed_ingest = input(
                    "Ingestion errors occurred. Examine .ingestion_errors and state.full_text.\n"
                    "Hit 'c' to confirm having read this message:\n\n\n"
                ).lower()
            
            if abort_failed_ingest == "y":
                print("Aborting ingestion. Please review the ingestion errors (returned below and accessible via .ingestion_errors) and the state.full_text object to see which papers were not ingested successfully.\n\n\n")
                return(self.ingestion_errors)
            else:
                pass # continue with ingestion despite errors

        # Create an ingestion status dataframe tracking ingestion success/failure linked by paper_id and question_id
        ingestion_status_df = pd.DataFrame({
            "paper_id": [os.path.splitext(os.path.basename(path))[0] for path in list_of_files],
            "question_id": [os.path.basename(os.path.dirname(path)) for path in list_of_files],
            "ingestion_status": ingestion_status, 
            "pages": list_of_papers_by_page
        })


        # Clean up working insights to reflect what came in mathcing an id, what was in there for which no file was found, and what came in that did not match an id
        working_insights = (
            ingestion_status_df.merge(working_insights, how="outer", on=["question_id", "paper_id"]) # Outer merge to make sure all the ingestion status records and all the insights records are included so we don't lose anyting
            .assign(ingestion_ids_matched = lambda x: np.where(x["question_text"].isna(), "import file does not match possible id", "match")) # First check if the question_text is na. If so, that means the ingested item did not match to a paper_id and question_id and therefore is unmatched
            .query("paper_id.notna()") # Filter any records for which paper_id is na, as this means there was not paper (usually a hangover from entering just questions to match with a folder of files - i.e. dump files in a folder and query them)
            .assign(ingestion_status=lambda x: np.where(x["ingestion_status"].isna(), 0, x["ingestion_status"])) # Set the ingestion status to 0 if it is na, meaning there was no paper ingested and thus no match
            .assign(ingestion_ids_matched=lambda x: np.where(x["ingestion_status"]==0, "existing id had no corresponding file", x["ingestion_ids_matched"])) # Check if the ingestion status is 0. If so, there was no paper ingested and thus no match - these are essentially docs that were not downloaded
        )

        # Get all the ids for which no file was found
        failed_id_matches = working_insights[working_insights["ingestion_ids_matched"] != "match"] # Get all the non-match ids: both files that came in without an id and those with an id for which no file came in
        self.failed_id_matches= failed_id_matches

        if failed_id_matches.shape[0] > 0:
            abort_failed_match = None
            while abort_failed_match not in ['y', 'n']:
                abort_failed_match = input(
                f"Warning: {failed_id_matches.shape[0]} paper(s) in the insights table "
                "did not have a matching file in the ingestion directory. This is not neccessarily an error, but if you want to be able to match " 
                "these papers to thier search terms and search engines later you will need to ensure the files are named correctly.\n\n"
                "If you are conducting a literature review this warning is likely relevant to you. If you are reading your own corpus, you can likely ignore this message.\n\n"
                "If you ignore this warning any paper ids that did not have a matching file will be deleted from state.insights. "  
                "You can look these up later by exploring the state.insights object created earlier in the pipeline.\n\n"
                "Do you wish to abort ingestion to review the failed id matches? (y/n):\n\n"
            ).lower()
            
            if abort_failed_match == 'y':
                print("Aborting ingestion. Please review the failed id matches (returned below and accessible via .failed_id_matches).\n\n")
                return(failed_id_matches)
            else:
                pass #continue with ingestion despite failed id matches


        # Identify all undownloaded files for the user to see what they are not getting. 
        # Note not tracking these is a design. This package manages reading. It has a module that helps with identifying papers to read, but it is the users responsibility to get the papers. 
        # So we record this error here but it is not persisted to state
        self.dropped_papers = working_insights[working_insights["ingestion_status"] == 0]

        # Get the all the ingested papers
        working_insights = working_insights[working_insights["ingestion_status"] == 1] # Filter to just the papers that were ingested successfully, as these are the ones we have insights for and can track through the pipeline.
        # Populate the full text state object
        full_text = (
            working_insights[["paper_id", "pages"]]
            .assign(full_text=lambda x: ["".join(pages) for pages in x["pages"]])
            .drop(columns=["pages"]) # Drop pages as they take up memory and are not needed, we have the full text now
        )
        # Set the full text as a state attribute
        self.state.full_text = full_text

        # Drop pages from insights as well as other fields from the lit retrieve module that are no longer needed:
        working_insights.drop(columns=["pages", "download_status", "messy_question_id", "messy_paper_id"], inplace=True)

        # Set as state attributes
        self.state.insights = working_insights

        if self.dropped_papers.shape[0] > 0: # Set as none on init, gets created if there are dropped papers
                print(
                    f"Warning: {self.dropped_papers.shape[0]} paper(s) were not downloaded from your original list. These papers are listed in the .dropped_papers attribute.\n"
                    "You can review these papers to see what was not ingested successfully, update and potentially re-ingest them, but they will not be included in the rest of the pipeline as we have no text to work with for these papers.\n\n"
                    )
                
        print("\nPaper ingestion complete")

    def _get_metadata_from_llm(self, paper_id: str, text: str) -> dict[str, Any]:
        """Call the LLM to extract metadata from the first three pages of a paper."""
        
        # Set variables for the call_chat_completion function from utils
        sys_prompt = Prompts().get_metadata()
        user_prompt = f"paper_id: {paper_id}\nTEXT:\n{text}"
        fallback = {
            "paper_id": paper_id,
            "paper_title": "NA",
            "paper_author": "NA",
            "paper_date": "NA"
        }
    
        response_dict = utils.call_chat_completion(llm_client=self.llm_client,
                                                    ai_model=self.ai_model,
                                                    sys_prompt=sys_prompt,
                                                    user_prompt=user_prompt,
                                                    return_json=True,
                                                    fall_back=fallback)

        # Validate keys
        required_keys = ["paper_id", "paper_title", "paper_author", "paper_date"]
        for key in required_keys:
            if key not in response_dict:
                raise KeyError(f"Metadata extraction failed: missing key '{key}'")

        # Clean paper_date
        paper_date = response_dict["paper_date"]
        if isinstance(paper_date, str):
            paper_date = paper_date.strip()
            if paper_date.upper() == "NA" or paper_date == "":
                response_dict["paper_date"] = pd.NA
            else:
                response_dict["paper_date"] = int(paper_date)
        else:
            response_dict["paper_date"] = paper_date

        return response_dict
    
    @staticmethod
    def _metadata_type_check(x, desired_type):
            # Missing or explicit NA
            if pd.isna(x):
                return pd.NA
            if isinstance(x, str):
                s = x.strip()
                if s.upper() == "NA" or s == "":
                    return pd.NA
                # Convert year-like strings when int requested
                if desired_type is int:
                    return int(s)  # assumes s is YYYY
                if desired_type is str:
                    return s

            # Non-string already correct type
            if desired_type is int and isinstance(x, int):
                return x
            if desired_type is str and isinstance(x, str):
                return x

            # If something unexpected comes back (e.g., list/dict), treat as missing
            return pd.NA
    
    def _populate_metadata(self, 
                           metadata_check_df: pd.DataFrame, # The dataframe containing the columns neccesary for the metadata check which is paper_id, paper_title, paper_author, paper_date, and full_text.
                           recovered_metadata_check: Optional[List[pd.DataFrame]] = None # The recoevered metadata from a previous run if the metadata check was interrupted and needs to be resumed. 
                           ) -> List[pd.DataFrame]:
        # If there is no recovered metadata passed to the function then start with an empty list, otherwise start with the recovered metadata
        if recovered_metadata_check is None:
            output = []
        else:
            output = recovered_metadata_check
            # Get the lenght of this list to see how many entries have been handled and drop them
            start = len(recovered_metadata_check) 
            metadata_check_df = metadata_check_df[start:] 

        # Now iterate either over the full df populating the empty list or over the partial dataframe populating the partially completed list
        for idx, row in metadata_check_df.iterrows():
            print(f"Checking metadata for paper {idx + 1} of {metadata_check_df.shape[0]}...")
            paper_id = row["paper_id"]
            text = row["full_text"][:5000] if row["full_text"] else ""
            metadata = self._get_metadata_from_llm(paper_id, text) # Get the metadat from the llm
            # Call the metadata type check function to ensure the metadata is in the correct format and handle any unexpected formats that may come back from the llm, such as lists or dicts, which we want to treat as missing values. This is important for ensuring the integrity of the metadata and avoiding issues later in the pipeline when we rely on this metadata for linking insights to papers and for any analyses that involve metadata.
            author = self._metadata_type_check(metadata.get("paper_author"), str)
            title  = self._metadata_type_check(metadata.get("paper_title"), str)
            year   = self._metadata_type_check(metadata.get("paper_date"), int)
            
            # create a new dataframe with the metadata pinned to paper id
            paper_meta_df = pd.DataFrame({
                "paper_id": [paper_id],
                "paper_title": [title],
                "paper_author": [author],
                "paper_date": [year]
            })

            output.append(paper_meta_df)
            
            # Save to pickle to allow for resume, make sure path exists
            os.makedirs(self.pickle_path, exist_ok=True)
            with open(os.path.join(self.pickle_path, "metadata_check.pkl"), "wb") as f:
                pickle.dump(output, f)

        return(output)

    def update_metadata(self) -> pd.DataFrame:
        """Get the metadata for every paper from the first 5000 chrs of the full text using the LLM and update state.insights with the metadata."""
        #Create the metadata check which is the dataframe containing the columns i need for the check
        metadata_check_df = (
            self.state.insights.copy()[["paper_id", "paper_title", "paper_author", "paper_date"]]
            .merge(self.state.full_text[["paper_id", "full_text"]],
                how="left",
                on=["paper_id"])
        )

        # Check whether a file exists which shows previoulsy completed meta data check results and ask the user what they want to do: rerun or resume

        if os.path.isfile(os.path.join(self.pickle_path, "metadata_check.pkl")):
            reload_meta_data = None
            while reload_meta_data not in ["1", "2"]:
                reload_meta_data = input(
                    "A metadata check pickle file was found. This indicates that a metadata check was previously run but may not have completed. \n"
                    "Do you want to: (pick the corresponding number)\n"
                    "  1 - Reload/resume the previous metadata check\n"
                    "  2 - Re-run the metadata check from the start\n"
                ).lower()

            # If they want to resume, get the data and run with recovered_metadata 
            if reload_meta_data == "1":
                with open(os.path.join(self.pickle_path, "metadata_check.pkl"), "rb") as f:
                    recovered_metadata = pickle.load(f)
                metadata_list = self._populate_metadata(metadata_check_df, recovered_metadata_check=recovered_metadata)
            # If they want to rerun, run the check from the start with an empty list for the output
            else:
                metadata_list = self._populate_metadata(metadata_check_df)
        
        # If the file does not exist, run the metadata check from the start with an empty list for the output
        else:
            metadata_list = self._populate_metadata(metadata_check_df)
         
        # Now process the list: concat to full metadata
        full_meta_data_df = pd.concat(metadata_list, ignore_index=True)

        # Merge back to state.insights. Drop the old metadata columns and replace with the new metadata from the llm. This ensures that any metadata that was missing or incorrect is updated with the llm response, while any existing correct metadata is retained. We merge on paper_id to ensure we are updating the correct records, and we validate one_to_one to ensure there are no duplicate paper_ids which would indicate an issue with the data integrity.
        updated_insights = (
            self.state.insights
            .drop(columns=["paper_title", "paper_author", "paper_date"])
            .merge(full_meta_data_df, how="left", on="paper_id", validate="one_to_one")
        )

        # Update the state.insights to now have the correct metadata. 
        # Note we don't save here as its not the end of the object and we have the pickle to handle recovery if we need to re run.
        self.state.insights = updated_insights

        return self.state.insights

    def chunk_papers(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        length_function=len,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False
    ) -> None:
        """Split full_text into nested chunks and flatten for downstream processing."""
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            is_separator_regex=is_separator_regex
        )

        full_text_list = self.state.full_text["full_text"].to_list()
        chunks_list: List[List[str]] = [text_splitter.split_text(text) for text in full_text_list]

        # Create the chunks state from the full_text state
        self.state.full_text["chunk_text"] = chunks_list
        self.state.chunks = self.state.full_text[["paper_id", "chunk_text"]].explode("chunk_text").reset_index(drop=True).copy()
        self.state.chunks["chunk_id"] = [f"chunk_{i+1}" for i in range(self.state.chunks.shape[0])]

        # Chunks from full_text as its now joined by paper and question id
        self.state.full_text.drop(columns=["chunk_text"], inplace=True)

        # Get chunk_id into insights
        temp_insights = self.state.insights.copy()
        self.state.insights = (
            self.state.chunks
            .drop(columns=["chunk_text"])
            .merge(temp_insights, how="left", on="paper_id")
        )

        # Save the updated state
        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "06_full_text_and_chunks"))

class Insights:
    def __init__(
        self,
        state: "QuestionState",
        llm_client: Any,
        ai_model: str,
        paper_context: str, 
        max_token_length: int = 100000,
        pickle_path: str = config.PICKLE_SAVE_LOCATION, 
        chunk_insights_pickle_file: str="chunk_insights.pkl", 
        meta_insights_pickle_file: str="meta_insights.pkl"
    ) -> None:
        """
        Class for extracting insights (both chunk-level and meta/paper-level) 
        from a corpus of academic papers and grey literature using an LLM.

        Args:
            state (QuestionState): 
                Container for all relevant state data including chunks, 
                full text, and insights tables.
            llm_client (Any): 
                Client instance for calling the LLM API (e.g. OpenAI client).
            ai_model (str): 
                Model name/ID to be used for completions.
        """
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.pickle_path: str = pickle_path
        os.makedirs(self.pickle_path, exist_ok=True)
        
        self.paper_context: str = paper_context
        self.chunk_insights_pickle_file = chunk_insights_pickle_file
        self.meta_insights_pickle_file = meta_insights_pickle_file
        self.max_token_length: int = max_token_length


        # Ensure state has all required columns before processing
        self.state = deepcopy(
            utils.validate_format(
            state=state,
            questions=None,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi"
            ],
            injected_required_cols=None
            )
        )

        self.state.enforce_canonical_question_text()

    

    def _generate_chunk_insights(self, insights: List = None) -> pd.DataFrame:
        """
        Generate research-question–level insights for each text chunk using the LLM.

        Each chunk is processed independently and evaluated against all defined
        research questions. For every chunk, the model returns zero or more
        insights per question. Even when no insights are identified, the chunk
        is recorded to ensure complete traceability and resumable execution.

        All generated insights are explicitly linked to:
            - chunk_id  (unique identifier for the text chunk)
            - paper_id  (source document identifier)

        The method supports resumable execution by accepting previously
        generated insight DataFrames and skipping already processed chunk_ids.

        Returns:
            pd.DataFrame:
                A consolidated insights table where each row represents a
                (question_id, paper_id, chunk_id, insight) record, merged
                with existing state metadata and assigned back to
                `self.state.insights`.

        # -------------------------------------------------------------
        # Recovery / Resume Handling
        #
        # This function can be called in two modes:
        #
        # 1. Fresh run:
        #    insights is None
        #    → No chunks have been processed yet.
        #
        # 2. Resume run:
        #    insights is a list of previously generated DataFrames,
        #    each containing chunk-level results that were written
        #    incrementally to pickle.
        #
        # We must defensively handle three distinct states:
        #
        #    insights is None   → brand new run
        #    insights == []     → valid recovery state but zero chunks processed
        #    insights has data  → resume and skip already-processed chunk_ids
        #
        # Even though under normal execution the pickle file should
        # never contain an empty list (because we append before dumping),
        # we explicitly support insights == [] to guard against:
        #   - manual injection
        #   - corrupted pickle files
        #   - future refactors
        #   - direct unit test calls
        #
        # The key invariant:
        #   remaining_chunks must ALWAYS be defined.
        # -------------------------------------------------------------
        """

        if insights is None:
            insights = []
            processed_chunks = []
        else:
            if len(insights) > 0:
                insights_df = pd.concat(insights).reset_index(drop=True)
                processed_chunks = insights_df["chunk_id"].unique().tolist()
            else:
                # Explicitly handle injected empty list
                processed_chunks = []

        remaining_chunks = self.state.chunks[
            ~self.state.chunks["chunk_id"].isin(processed_chunks)
        ]

        # Generate a list of all the research questions with ids in the form <rq_id>: <rq_text> for the llm to consdier against each chunk
        rqs_ids = [f"{row['question_id']}: {row['question_text']}" for _, row in self.state.questions.iterrows()]
        rqs_ids_str = "\n".join(rqs_ids)

        temp_state_insights = self.state.insights.copy()

        # Merge chunk text with metadata (author, date, etc.)
        temp_state_insights: pd.DataFrame = remaining_chunks.merge(
            temp_state_insights[["paper_id", "question_text", "paper_author", "paper_date"]],
            how="left",
            on=["paper_id"]
        )

        # Iterate over each chunk
        for idx, (df_index, row) in enumerate(temp_state_insights.iterrows()):
            print(f"Processing chunk {len(processed_chunks) + idx + 1} of {temp_state_insights.shape[0] + len(processed_chunks)}...")

            # Extract fields from row
            paper_id: str = row["paper_id"]
            chunk_text = row["chunk_text"] if pd.notna(row["chunk_text"]) else ""
            chunk_id: str = str(row["chunk_id"])

            # Generate the citation accounting for NA values in authors and date
            authors = row["paper_author"]
            if isinstance(authors, (list, np.ndarray)):
                citation = " ".join(authors)
            elif pd.isna(authors):
                citation = ""
            else:
                citation = str(authors)
            date = row["paper_date"] if not pd.isna(row["paper_date"]) else ""
            citation = f"{citation} {date}"
           

            # Build prompts
            sys_prompt: str = Prompts().gen_chunk_insights(paper_context=self.paper_context)
            user_prompt: str = (
                f"RESEARCH QUESTIONS:\n{rqs_ids_str}\n\n"
                f"TEXT CHUNK:\n{chunk_text} - {citation}\n"
            )

            fall_back = {
                "results": {}
            }

            response_dict = utils.call_chat_completion(ai_model = self.ai_model,
                                 llm_client = self.llm_client,
                                 sys_prompt = sys_prompt,
                                 user_prompt = user_prompt,
                                 return_json = True, 
                                 fall_back=fall_back)
            
            # Turn the response into a df with columns question_id and insight, where each row is a different insight, and the insights are the insights for that question id that were extracted from the chunk. This will make it easier to merge into the state later.
            results = response_dict.get("results", {})
            
            response_df = (
                pd.DataFrame(results.items(), columns=["question_id", "insight"])
                .explode("insight") #explode on insights to make tidy data from the lists
                .reset_index(drop=True)
                           )
            # If the response is empty (no insights were found) add a row with the chunk id and paper id
            if response_df.empty:
                response_df = pd.DataFrame([{
                    "question_id": pd.NA,
                    "insight": pd.NA, 
                    "chunk_id": chunk_id,
                    "paper_id": paper_id
                }])
            # Otherwise just add the chunk id and paper id to the response df
            else: 
                # Add paper_id and chunk_id to the response_df 
                response_df["chunk_id"] = chunk_id
                response_df["paper_id"] = paper_id

            # Append to insights list
            insights.append(response_df)
            with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "wb") as f:
                pickle.dump(insights, f)

        # Convert insights list to DataFrame
        print("Converting insights to DataFrame and merging into state...")
        insights_complete_df: pd.DataFrame = pd.concat(insights).reset_index(drop=True) if insights else pd.DataFrame(columns=["question_id", "insight", "chunk_id", "paper_id"])
        
        # Merge into global insights table - first drop existing insight columns if present (these can be created by previous runs of recover_chunk_insights_generation)
        base_insights = (
            self.state.insights
            .drop(columns=["insight"], errors="ignore") # drop the metadata columns if they exist as we will merge them back in from the state later, but ignore if they don't exist as this function can be run multiple times and they will only be there after the first run
        )
        
        # Now we merge this chunk insights with all the insights data and metadata. 
        # Notably we drop question_id from the state.insights as previously if papers wwere not associated with a question when importing them, the question_id was NA
        # Now we have all the chunks and thier insights associated with a question_id thus this becomes the primary df
        working_insights_df = (
            insights_complete_df
            .merge(
                base_insights.drop(columns=["question_id"], errors="ignore"), # So that we don't duplicate question_id columns
                how="left",
                on=["paper_id", "chunk_id"]
                ))
        
        # Add insight_ids
        mask = working_insights_df["insight"].notna()

        # Create numeric ids only where valid
        working_insights_df["insight_id"] = pd.NA
        working_insights_df.loc[mask, "insight_id"] = range(1, mask.sum() + 1)

        # Explicitly transform only valid rows to final string form
        working_insights_df.loc[mask, "insight_id"] = (
            "chunk_insight_" +
            working_insights_df.loc[mask, "insight_id"].astype(int).astype(str)
        )

        # Assign to state
        self.state.insights = working_insights_df

        # Note i don't save here as i only save at the end of the class's operations. This is cleaner for the user. 
        # Also since i have the recover function in place, once the insights are generated for this class it should be quick to recreate to this point
        return self.state.insights
    
    
    def _recover_chunk_insights_generation(self):
        print("Opening pickle file to recover chunk insights generation...")
        with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "rb") as f:
            recover_chunk_insights = pickle.load(f)
        
        self._generate_chunk_insights(insights=recover_chunk_insights)


    def get_chunk_insights(self) -> pd.DataFrame:
        if os.path.exists(os.path.join(self.pickle_path, self.chunk_insights_pickle_file)):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Insights already exist on file at {os.path.join(self.pickle_path, self.chunk_insights_pickle_file)}. "
                    "Do you wish to recover/restore or regenerate insights?\n"
                    "Hit 'r' to recover/restore from file, or 'n' to generate new insights (this will overwrite existing file):\n"
                    ).lower()
            if recover == 'r':
                self._recover_chunk_insights_generation()
            else:
                print("Overwriting existing chunk insights pickle file...")
                self._generate_chunk_insights()
        else:
            self._generate_chunk_insights()


    def _prepare_meta_insights_df(self) -> pd.DataFrame:
        """
        Generate the dataframe 'meta-insights'. This is compiled once to allow for easy passing to the LLM to extract meta-insights — arguments that span multiple chunks within 
        the same paper. Creating this single dataframe allows us to manage resume if the user has to abort the generation process.
        The dataframe includes the paper_id, the full_text (broken into chunks that max the context window), the chunk insights for the paper and meatadata. 
        All data is tidy with the full_text chunk the most granular level 


        Returns:
            pd.DataFrame: DataFrame of meta-insights 

        Raises:
            ValueError: If chunk insights do not exist prior to running.
        """
        # Must run chunk insights first
        if "insight" not in self.state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        rqs = self.state.questions
        # Create the final list of paper that we will populate as we develop all the data for checking for meta insights for each paper
        list_of_papers = []
        # Get the full text and paper id
        for _, row in self.state.full_text.iterrows():
            paper_id = row["paper_id"]
            paper_content = row["full_text"]
            # Check that the whole paper fits in the model context window, if not break into chunks and process separately 
            # (this is a bit of a hack but it allows us to at least get some meta insights from papers that exceed the context window, 
            # which is likely to be the case for many academic papers with the full text included)
            token_count = self.estimate_tokens(paper_content, self.ai_model)
            if token_count > self.max_token_length:
                paper_content_list = self.string_breaker(paper_content, max_token_length=self.max_token_length)
            else:
                paper_content_list = [paper_content]

            # Get the chunk insights for the paper id and the question_id
            for rqid in rqs["question_id"].to_list():
                paper_question_chunk_insights = self.state.insights[
                    (self.state.insights["paper_id"] == paper_id) & (self.state.insights["question_id"] == rqid)
                ]["insight"].dropna().tolist()
                paper_question_chunk_insights = "\n".join(paper_question_chunk_insights) if paper_question_chunk_insights else ""
            
                # Get the metadata for the paper id
                paper_metadata_df = self.state.insights[self.state.insights["paper_id"] == paper_id][["paper_author", "paper_title", "paper_date"]].drop_duplicates()
                author = paper_metadata_df["paper_author"].iloc[0] if not paper_metadata_df.empty and "paper_author" in paper_metadata_df.columns else pd.NA
                if isinstance(author, list):
                        author_str = ", ".join(author)
                elif pd.isna(author):
                    author_str = ""
                else:
                    author_str = str(author)
                date = paper_metadata_df["paper_date"].iloc[0] if not paper_metadata_df.empty and "paper_date" in paper_metadata_df.columns else pd.NA
                date_str = "" if pd.isna(date) else str(date)
                title = paper_metadata_df["paper_title"].iloc[0] if not paper_metadata_df.empty and "paper_title" in paper_metadata_df.columns else pd.NA
                title_str = "" if pd.isna(title) else str(title)
                metadata = f"{author_str}, {date_str}, {title_str}"

                # Now build the dataframe that we can call against the LLM and that we can use to determine resume points
                meta_insight_check_df = pd.DataFrame({
                    "paper_id": [paper_id] * len(paper_content_list),
                    "question_id": [rqid] * len(paper_content_list),
                    "paper_content": paper_content_list,
                    "paper_chunk_insights": [paper_question_chunk_insights] * len(paper_content_list),
                    "metadata": [metadata] * len(paper_content_list),
                })

                list_of_papers.append(meta_insight_check_df)

        # Concat all the papers in the list
        meta_insight_check_df = pd.concat(list_of_papers).reset_index(drop=True)    

        # Create an id for full_content chunks both for resuming and traceability
        meta_insight_check_df["content_chunk_id"] = [f"meta_chunk_{i+1}" for i in range(meta_insight_check_df.shape[0])]

        # Update the state.chunks with the meta chunks and their ids
        temp_chunks = self.state.chunks.copy()
        # Check if the meta_chunks were already created in a previous run. If so remove them from the state object before we concat so that they don't get duplicated. 
        if temp_chunks["chunk_id"].str.startswith("meta_chunk_").any():
            temp_chunks = temp_chunks[~temp_chunks["chunk_id"].str.startswith("meta_chunk_")]
        
        self.state.chunks = pd.concat([
            meta_insight_check_df[["paper_id", "paper_content", "content_chunk_id"]].rename(columns={"paper_content": "chunk_text", "content_chunk_id": "chunk_id"}),
            temp_chunks
            ])

        return(meta_insight_check_df)
        

    def get_meta_insights(self) -> pd.DataFrame:
        """
        Generate 'meta-insights' — arguments that span multiple chunks within 
        the same paper. Each paper is processed once, combining all chunk insights 
        and the full text.

        Returns:
            pd.DataFrame: An updated version of state.insights which has the meta-insights appended.

        Raises:
            ValueError: If chunk insights do not exist prior to running.
        """
        # Must run chunk insights first
        if "insight" not in self.state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        # Check if there is a pickle file with previously generated meta_insights and ask the user if they want to recover or regenerate
        meta_insights = None
        if os.path.exists(os.path.join(self.pickle_path, self.meta_insights_pickle_file)):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Meta-insights already exist on file at {os.path.join(self.pickle_path, self.meta_insights_pickle_file)}. "
                    "Do you wish to recover/restore or regenerate meta-insights?\n"
                    "Hit 'r' to recover/restore from file, or 'n' to generate new meta-insights (this will overwrite existing file):\n"
                    ).lower()
            if recover == 'r':
                print("Recovering meta-insights from file...")
                with open(os.path.join(self.pickle_path, self.meta_insights_pickle_file), "rb") as f:
                    meta_insights = pickle.load(f)
            else:
                # Clean up any meta insights that have been generated in self.state.insights to avoid duplication when we generate new ones.
                self.state.insights = self.state.insights[
                    ~self.state.insights["insight_id"].astype(str).str.startswith("meta_insight_")
                    ]
                meta_insights = None

        # If meta insights exists (its a list of dicts), so we concat them to get the ids of all the processed content chunks so we can drop them from the meta_insight_check_df and only process the remaining content chunks. 
        if meta_insights is not None:
            meta_insights_run_df = pd.concat(meta_insights).reset_index(drop=True)
            processed_meta_chunks = meta_insights_run_df["content_chunk_id"].unique().tolist()
        else: 
            processed_meta_chunks = [] #if it doesn't exist its empty so we will exclude nothing

        # Drop the processed content chunks
        meta_insight_check_df = self._prepare_meta_insights_df()
        meta_insight_check_df = meta_insight_check_df[~meta_insight_check_df["content_chunk_id"].isin(processed_meta_chunks)]

        # Now prepare to pass to the LLM for checking
        meta_insights_df_lst = [] if meta_insights is None else meta_insights
        for idx, row in meta_insight_check_df.iterrows():
            print(f"Processing meta-insights for content piece {idx + len(processed_meta_chunks) + 1} of {meta_insight_check_df.shape[0] + len(processed_meta_chunks)}...")

            # Extract fields from row
            paper_id: str = row["paper_id"]
            rq_id = row["question_id"]
            rq_text = self.state.questions[self.state.questions["question_id"] == rq_id]["question_text"].iloc[0] if rq_id in self.state.questions["question_id"].tolist() else ""
            rq = f"{rq_id}: {rq_text}"
            other_rqs = self.state.questions[self.state.questions["question_id"] != rq_id]
            other_rqs_str = "\n".join([f"{row['question_id']}: {row['question_text']}" for _, row in other_rqs.iterrows()])
            paper_content: str = row["paper_content"] if pd.notna(row["paper_content"]) else ""
            paper_chunk_insights: str = row["paper_chunk_insights"] if pd.notna(row["paper_chunk_insights"]) else ""
            metadata: str = row["metadata"] if pd.notna(row["metadata"]) else ""
            content_chunk_id: str = row["content_chunk_id"]


            # Build prompts
            sys_prompt: str = Prompts().gen_meta_insights(paper_context=self.paper_context)
            user_prompt: str = (
                f"SPECIFIC RESEARCH QUESTION FOR CONSIDERATION:\n{rq}\n"
                f"PAPER METADATA:\n{metadata}\n"
                f"PAPER TEXT:\n{paper_content}\n"
                f"EXISTING CHUNK INSIGHTS:\n{paper_chunk_insights}\n\n"
                f"OTHER RESEARCH QUESTIONS IN THE REVIEW (context only):\n{other_rqs_str}\n\n"
            )

            fall_back = {
                "results": {}
            }

            response_dict = utils.call_chat_completion(ai_model = self.ai_model,
                                llm_client = self.llm_client,
                                sys_prompt = sys_prompt,
                                user_prompt = user_prompt,
                                return_json = True, 
                                fall_back=fall_back)
            
            results = pd.DataFrame(response_dict["results"])

            # If the response is empty use an empty list otherwise get the list of results from results
            meta_list = results["meta_insight"].tolist() if not results.empty else []

            meta_insight_df = pd.DataFrame({
                "paper_id": [paper_id],
                "question_id": [rq_id],
                "content_chunk_id": [content_chunk_id],
                "meta_insight": [meta_list]
            })

            meta_insights_df_lst.append(meta_insight_df)

            with open(os.path.join(self.pickle_path, self.meta_insights_pickle_file), "wb") as f:
                pickle.dump(meta_insights_df_lst, f)


        # Now join up the list of dataframes to get a single value
        meta_insights_complete_df: pd.DataFrame = pd.concat(meta_insights_df_lst).reset_index(drop=True) if meta_insights_df_lst else pd.DataFrame(columns=["paper_id", "question_id", "content_chunk_id", "meta_insight"])
        # Explode on meta_insight (currently a list for each content chunk)
        meta_insights_complete_df = meta_insights_complete_df.explode("meta_insight").reset_index(drop=True) 

        # Now we want to append to self.state.insights (which already has the chunk level insights)
        # First rename meta_insight to insight to match .state.insights, i also rename content_chunk_id to chunk_id to match the state and ensure the meta_chunks have an ID of thier own. 
        meta_insights_complete_df.rename(columns={"meta_insight": "insight", 
                                                  "content_chunk_id": "chunk_id"}, 
                                                  inplace=True)
        #Then add insight ids to meta insights to distinguish them from chunk insights and to have a unique id for each insight which we can use for traceability 
        # First create mask so that we only iterate insight_id count for insights, not for NA insights
        mask = meta_insights_complete_df["insight"].notna()

        # Create numeric ids only where valid
        meta_insights_complete_df["insight_id"] = pd.NA
        meta_insights_complete_df.loc[mask, "insight_id"] = range(1, mask.sum() + 1)

        # Explicitly transform only valid rows to final string form
        meta_insights_complete_df.loc[mask, "insight_id"] = (
            "meta_insight_" +
            meta_insights_complete_df.loc[mask, "insight_id"].astype(int).astype(str)
        )

        # Then manipulate state.insights so that we can merge with meta_insights to get a complete df that we can later append to insights
        temp_insights = self.state.insights.copy()
        temp_insights = (
            temp_insights
            .drop(columns=["question_id", "question_text", "insight_id", "insight", "chunk_id"]) #Drop these to avoid conflicts
            .drop_duplicates(subset=["paper_id"])
            ) 
        meta_insights_complete = meta_insights_complete_df.merge(
            temp_insights, 
            on="paper_id", 
            how="left"
            )
        # Now we have the meta insights with all the metadata and we can append to the state insights
        self.state.insights = pd.concat([self.state.insights, meta_insights_complete], ignore_index=True)
        # Save to parquet and return
        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "07_insights"))
        return self.state.insights

      
    @staticmethod
    def ensure_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if pd.isna(x):
            return []
        return [x]  # fallback for any other type
    
    @staticmethod
    def estimate_tokens(text, model):
        """Estimate token count for a given text and model using tiktoken."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @staticmethod
    def string_breaker(text, max_token_length):
        """Break a long string into a list of strings each less than max length."""
        max_length = max_token_length * 0.75
        words = text.split()
        current_chunk = ""
        chunks = []
        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                current_chunk += (" " if current_chunk else "") + word
            else:
                chunks.append(current_chunk)
                current_chunk = word
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
    # def recover_meta_insights_generation(self):
    #     print("Opening pickle file to recover meta insights generation...")
    #     with open(os.path.join(self.pickle_path, self.meta_insights_pickle_file), "rb") as f:
    #         recover_meta_insights = pickle.load(f)
        
    #     start = len(recover_meta_insights)
    #     print(f"Resuming meta insights generation from paper {start}...")
    #     self.get_meta_insights()
    
class Clustering:
    """
    Manage embedding, dimensionality reduction, clustering, and cluster evaluation
    for insights associated with research questions, while safely handling empty insights.
    """

    def __init__(
        self,
        state: QuestionState,
        llm_client: Any,
        embedding_model: str,
        embedding_dims: int = 1024,
        embeddings_pickle_path: str = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")
    ):
        self.state = deepcopy(
            utils.validate_format(
            state=state,
            questions = None,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi", "chunk_id", "insight"
            ],
            injected_required_cols=None
            )
        )

        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.embeddings_pickle_path = embeddings_pickle_path

        self.valid_embeddings_df: pd.DataFrame = self._gen_valid_embeddings_df()
        self.insight_embeddings_array: np.ndarray = np.array([])
        self.reduced_insight_embeddings_array: np.ndarray = np.array([])
        self.cum_prop_cluster: pd.DataFrame = pd.DataFrame()

    
    @staticmethod
    def _strip_citation_parentheticals(text: str) -> str:
        """
        Remove citation-style parentheticals to reduce embedding bias.
        """

        if not isinstance(text, str):
            return ""

        # 1. Remove parentheses containing a 4-digit year
        text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)

        # 2. Remove parentheses that look like author lists (capitalized names)
        text = re.sub(r"\(([A-Z][a-zA-Z]+(?:\s+(?:and|et al\.|,)?\s*[A-Z][a-zA-Z]+)+)\)", "", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _gen_valid_embeddings_df(self):
        """
        Get the DataFrame of insights that are non-empty after stripping
        citation parentheticals.
        Updates self.state.insights with a new column 'no_author_insight_string'.
        Returns: pd.DataFrame
        """

        # Strip citation-style parentheticals from the insight text
        self.state.insights["no_author_insight_string"] = (
            self.state.insights["insight"]
            .astype(str)
            .apply(self._strip_citation_parentheticals)
        )

        # Keep only non-empty strings
        out = self.state.insights[
            self.state.insights["no_author_insight_string"].str.strip() != ""
        ].copy()

        return out
    
    def embed_insights(self) -> np.ndarray:
        """
        Generate embeddings for non-empty insights only.
        Returns:
            np.ndarray: 2D array of embeddings for valid insights.
        """
        # Check if embeddings pickle exists
        if os.path.exists(self.embeddings_pickle_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Embeddings pickle found at '{self.embeddings_pickle_path}'. "
                    "Do you wish to reload them or regenerate embeddings?\n"
                    "Hit 'n' to generate new embeddings (this will overwrite existing pickle), or 'r' to reload from file:\n"
                ).lower()
            if recover == 'r':
                self._load_embeddings()
                return(self.insight_embeddings_array)
            else:
                print("Overwriting existing embeddings pickle...")
                
        # Generate embeddings for valid insights only
        insight_embeddings = []
        for idx, insight in enumerate(self.valid_embeddings_df["no_author_insight_string"], start=1):
            print(f"Embedding insight {idx} of {self.valid_embeddings_df.shape[0]}")
            response = self.llm_client.embeddings.create(
                input=insight,
                model=self.embedding_model,
                dimensions=self.embedding_dims
            )
            insight_embeddings.append(response.data[0].embedding)


        self.insight_embeddings_array = np.vstack(insight_embeddings)
        self.valid_embeddings_df["full_insight_embedding"] = [emb.tolist() for emb in self.insight_embeddings_array]

        self._save_embeddings()  # safe pickle save
        return self.insight_embeddings_array

    def _save_embeddings(self):
        """Save embeddings safely, creating folder if it does not exist."""
        os.makedirs(os.path.dirname(self.embeddings_pickle_path), exist_ok=True)
        with open(self.embeddings_pickle_path, "wb") as f:
            pickle.dump(self.insight_embeddings_array, f)
        print(f"Embeddings safely saved to '{self.embeddings_pickle_path}'.")

    def _load_embeddings(self):
        """Load embeddings safely if the pickle exists."""
        if not os.path.exists(self.embeddings_pickle_path):
            raise FileNotFoundError(f"No embeddings pickle found at {self.embeddings_pickle_path}")
        with open(self.embeddings_pickle_path, "rb") as f:
            data = pickle.load(f)
        self.insight_embeddings_array = data
        print("Linking embeddings back to valid embeddings DataFrame...")
        self.valid_embeddings_df["full_insight_embedding"] = [emb.tolist() for emb in self.insight_embeddings_array]    
        print(f"Embeddings loaded from '{self.embeddings_pickle_path}'.")
        return self.insight_embeddings_array

    def reduce_dimensions(
        self, full_embeddings: np.array = None, n_neighbors: int = 15, min_dist: float = 0.25, n_components: int = 10,
        metric: str = "cosine", random_state: int = 42, update_attributes: bool = True
    ) -> np.ndarray:
        
        # See if full_embeddings were provided, otherwise use the class's insight embeddings (this is done to make other methods in the class more explicit)
        if full_embeddings is None:
            full_embeddings = self.insight_embeddings_array

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )

        reduced_embeddings = reducer.fit_transform(full_embeddings)

        if update_attributes:
            # create some attributes for downstream use - skip this when tuning params
            self.reduced_insight_embeddings_array = reduced_embeddings
            self.valid_embeddings_df["reduced_insight_embedding"] = [row.tolist() for row in self.reduced_insight_embeddings_array]
            self.rq_valid_embeddings_dfs = {
                rq: self.valid_embeddings_df[self.valid_embeddings_df["question_id"] == rq].copy() for rq in self.valid_embeddings_df["question_id"].unique()
            }

        return reduced_embeddings

    def calc_silhouette(self, reduced_embeddings: np.array = None, rq_exclude: list[str] = None) -> float:
        """
        Calculate the silhouette score for the current reduced embeddings,
        using research question IDs as cluster labels.

        Args:
            rq_exclude (list[str], optional): 
                List of question_id strings to exclude from the silhouette calculation.
                If provided, any rows with question_id in this list will be excluded.

        Returns:
            float: The silhouette score (higher is better cluster separation).
        """

        if reduced_embeddings is None:
            reduced_embeddings = self.reduced_insight_embeddings_array

        sil_df = self.valid_embeddings_df.copy()
        sil_df["reduced_insight_embedding"] = [row.tolist() for row in reduced_embeddings]

        if rq_exclude:
            # Exclude any rows where question_id is in the rq_exclude list
            sil_df = sil_df[~sil_df["question_id"].isin(rq_exclude)]

        score = silhouette_score(
            X=np.vstack(sil_df["reduced_insight_embedding"].to_list()),
            labels=sil_df["question_id"].to_numpy(),
            metric="euclidean"
        )
        print(f"Silhouette score: {score}")
        return score

    def tune_umap_params(
        self,
        n_neighbors_list: list[int] = [5, 15, 30, 50, 75, 100],
        min_dist_list: list[float] = [0.0, 0.1, 0.2, 0.5],
        n_components_list: list[int] = [5, 10, 20],
        metric_list: list[str] = ["cosine", "euclidean"],
        rq_exclude: list[str] = None
    ) -> None:
        """
        Grid search over UMAP dimensionality reduction parameters, evaluating each
        combination using silhouette score (optionally excluding certain questions).

        IMPORTANT: This tuning is being done to check how well the insights cluster according
        to the research questions they were derived for. This is intended as a proxy for their
        ability to identify meaningful semantic structure and therefore the likelihood they can
        generalize to new, unseen data (the clusters within research questions). So this function
        runs over all the research questions. It is not run per question.

        Args:
            n_neighbors_list (list[int]): List of UMAP n_neighbors values to try.
            min_dist_list (list[float]): List of UMAP min_dist values to try.
            n_components_list (list[int]): List of UMAP n_components (output dims) to try.
            metric_list (list[str]): List of UMAP distance metrics to try.
            rq_exclude (list[str], optional): 
                List of question_id strings to exclude from silhouette scoring.
                Useful for ignoring questions that drive overlap or noise.

        Returns:
            None. Results are stored in self.umap_param_tuning_results as a DataFrame.
        """
        results = []
        param_grid = list(itertools.product(n_neighbors_list, min_dist_list, n_components_list, metric_list))
        total_runs = len(param_grid)

        for run_num, (n_neighbors, min_dist, n_components, metric) in enumerate(param_grid, start=1):
            print(f"Run {run_num} of {total_runs}: n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components}, metric={metric}")
            reduced_embeddings = self.reduce_dimensions(
                full_embeddings=self.insight_embeddings_array,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric, 
                update_attributes=False  # do not store these temp reductions
            )
            # Calculate silhouette score, optionally excluding specified questions
            score = self.calc_silhouette(reduced_embeddings=reduced_embeddings, rq_exclude=rq_exclude)
            results.append({
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "n_components": n_components,
                "metric": metric,
                "silhouette_score": score
            })

        # Convert results to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        self.umap_param_tuning_results = results_df.sort_values("silhouette_score", ascending=False)
        print(self.umap_param_tuning_results)

    @staticmethod
    def cluster(embedding_matrix, min_cluster_size: int = 5, metric: str = "euclidean", cluster_selection_method: str = "eom"):
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )

        cluster_labels = clusterer.fit_predict(embedding_matrix)
        cluster_probs = clusterer.probabilities_

        return cluster_labels, cluster_probs
    
    @staticmethod
    def calc_davies_bouldain_score(embeddings_matrix, cluster_labels):
        mask = cluster_labels != -1
        num_outliers = np.sum(~mask)
        filtered_embeddings = embeddings_matrix[mask]
        filtered_labels = cluster_labels[mask]
        if len(set(filtered_labels)) < 2:
            return(pd.NA, num_outliers)
        db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
        return db_score, num_outliers

    def tune_hdbscan_params(self,
                            min_cluster_sizes: list[int] = [5, 10, 15, 20],
                            metrics: list[str] = ["euclidean", "manhattan"],
                            cluster_selection_methods: list[str] = ["eom", "leaf"]
                            ) -> None:
      
        param_grid = list(itertools.product(min_cluster_sizes, metrics, cluster_selection_methods))
        rqs = self.valid_embeddings_df["question_id"].unique()
        total_runs = len(param_grid) * len(rqs)
        data = [self.rq_valid_embeddings_dfs[rq] for rq in rqs]
        results = []
        for idx, (d, rq) in enumerate(zip(data, rqs)):
            print(f"Tuning HDBSCAN for {rq}...(run {idx + 1} of {total_runs})")
            embeddings_matrix = np.vstack(d["reduced_insight_embedding"].to_list())
            for min_cluster_size, metric, cluster_selection_method in param_grid:

                cluster_labels, _ = self.cluster(embeddings_matrix, min_cluster_size=min_cluster_size, metric=metric, cluster_selection_method=cluster_selection_method)
                db_score, num_outliers = self.calc_davies_bouldain_score(embeddings_matrix, cluster_labels)
                results.append({
                    "question_id": rq,
                    "min_cluster_size": min_cluster_size,
                    "metric": metric,
                    "cluster_selection_method": cluster_selection_method,
                    "db_score": db_score,
                    "num_outliers": num_outliers
                    })

        results_df = pd.DataFrame(results)
        self.hdbscan_tuning_results = results_df.sort_values(["question_id","db_score"], ascending=True)
        print(self.hdbscan_tuning_results)

    def generate_clusters(self, clustering_param_dict: dict) -> pd.DataFrame:
        """
        Generate clusters for each research question using HDBSCAN.
        Updates self.state.insights with cluster labels and probabilities.

        Args:
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            metric (str): Distance metric for HDBSCAN.
            cluster_selection_method (str): Cluster selection method for HDBSCAN.
        """
        
        rqs = self.valid_embeddings_df["question_id"].unique()
        # Check if clustering_param_dict has entries for all rqs
        if len(clustering_param_dict) != len(rqs):
            use_default = None
            while use_default not in ['y', 'n']:
                use_default = input(
                    f"You did not enter specific clustering parameters for each research question. "
                    "Do you want to use default parameters? (y/n): "
                ).lower()
                if use_default == 'n':
                    raise KeyboardInterrupt("Please rerun and provide clustering parameters for each research question.")
                else:
                    params = [clustering_param_dict.get(rq, {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"}) for rq in rqs]
        else:
            params = [clustering_param_dict[rq] for rq in rqs]
        
        data = [self.rq_valid_embeddings_dfs[rq] for rq in rqs]

        offset = 0  # Initialize offset for cluster label adjustment

        for d, rq, param in zip(data, rqs, params):
            embeddings_matrix = np.vstack(d["reduced_insight_embedding"].to_list())
            cluster_labels, cluster_probs = self.cluster(
                embedding_matrix=embeddings_matrix,
                min_cluster_size=param["min_cluster_size"],
                metric=param["metric"],
                cluster_selection_method=param["cluster_selection_method"]
            )

            d["cluster"] = cluster_labels
            d["cluster_prob"] = cluster_probs
            # Translate cluster labels stating from 1, with 1 being the largest 
            # 1. Get cluster sizes (excluding -1)
            cluster_sizes = d[d["cluster"] != -1].groupby("cluster").size().sort_values(ascending=False)
            # 2. Map old cluster labels to new ones (largest=1, next=2, etc.)
            label_map = {old: i+1+offset for i, old in enumerate(cluster_sizes.index)}
            # 3. Apply mapping, keep -1 as is
            d["cluster"] = d["cluster"].apply(lambda x: label_map[x] if x in label_map else -1)
            # 4. Update offset for next DataFrame
            if cluster_sizes.size > 0:
                offset = max(label_map.values())

        summary_df = [self.make_cum_prop_cluster_table(d) for d in data]
        summary_df = [df.assign(question_id=rq) for df, rq in zip(summary_df, rqs)]
        self.cum_prop_cluster = pd.concat(summary_df)
        clustered_df = pd.concat(data)

        self.state.insights = self.state.insights.merge(
            clustered_df[["question_id", "paper_id", "chunk_id", "insight_id", "cluster", "cluster_prob", "full_insight_embedding", "reduced_insight_embedding"]],
            on=["question_id", "paper_id", "chunk_id", "insight_id"],
            how="left"
        )

        return self.state.insights

    @staticmethod
    def make_cum_prop_cluster_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize the DataFrame by cluster, showing count, proportion, and cumulative proportion.
        Outliers (cluster == -1) are moved to the end.
        """
        # Exclude rows with missing cluster
        df = df.dropna(subset=["cluster"]).copy()

        # Count size of each cluster (excluding outliers)
        main = df[df["cluster"] != -1].groupby("cluster").size().sort_values(ascending=False)
        # Relabel clusters so largest is 1, next is 2, etc.
        label_map = {old: i+1 for i, old in enumerate(main.index)}
        df["cluster"] = df["cluster"].apply(lambda x: label_map[x] if x in label_map else -1)

        # Count again with new labels (including outliers)
        summary = (
            df.groupby("cluster")
            .size()
            .reset_index(name="count")
            .sort_values(["cluster"], key=lambda col: col.where(col != -1, 999))
            .reset_index(drop=True)
        )

        # Calculate proportion and cumulative proportion
        summary["prop"] = summary["count"] / summary["count"].sum()
        summary["cum_prop"] = summary["prop"].cumsum()

        # Move outlier (-1) to the end
        main_clusters = summary[summary["cluster"] != -1]
        outliers = summary[summary["cluster"] == -1]
        summary = pd.concat([main_clusters, outliers], ignore_index=True)

        return summary


    # def generate_clusters(
    #     self, min_cluster_size: int = 5, metric: str = "euclidean", cluster_selection_method: str = "eom"
    # ) -> pd.DataFrame:
    #     if self.reduced_insight_embeddings_array.size == 0:
    #         raise ValueError("Reduced embeddings not available. Run .embed_insights() and .reduce_dimensions() first.")

    #     clusterer = hdbscan.HDBSCAN(
    #         min_cluster_size=min_cluster_size,
    #         metric=metric,
    #         cluster_selection_method=cluster_selection_method
    #     )

    #     self.valid_embeddings_df["reduced_insight_embeddings"] = [row.tolist() for row in self.reduced_insight_embeddings_array]

    #     clustered_dfs = []
    #     for rq in self.valid_embeddings_df["question_id"].unique():
    #         print(f"Generating clusters for {rq}...")
    #         rq_df = self.valid_embeddings_df[self.valid_embeddings_df["question_id"] == rq].copy()
    #         embeddings_matrix = np.vstack(rq_df["reduced_insight_embeddings"].to_list())
    #         cluster_labels = clusterer.fit_predict(embeddings_matrix)
    #         cluster_probs = clusterer.probabilities_

    #         rq_df["cluster"] = cluster_labels
    #         rq_df["cluster_prob"] = cluster_probs
    #         clustered_dfs.append(rq_df)   

    #     clustered_df = pd.concat(clustered_dfs)

    #     # In case the user is re-running generate clusters to adjust parameters, remove the old cluster assignments
    #     for col in ["cluster", "cluster_prob"]:
    #         if col in self.state.insights.columns:
    #             self.state.insights.drop(columns=[col], inplace=True)   

    #     self.state.insights = self.state.insights.merge(
    #         clustered_df[["question_id", "paper_id", "chunk_id", "cluster", "cluster_prob"]],
    #         on=["question_id", "paper_id", "chunk_id"],
    #         how="left"
    #     )

    #     # 1. Calculate counts and prop per question_id and cluster
    #     cum_prop_cluster = (
    #         self.state.insights.dropna(subset=["cluster"])
    #         .groupby(["question_id", "cluster"])
    #         .size()
    #         .reset_index(name="count")
    #     )

    #     # 2. Calculate proportions within each question_id
    #     cum_prop_cluster["prop"] = cum_prop_cluster.groupby("question_id")["count"].transform(lambda x: x / x.sum())

    #     # 3. Move -1 to the end and calculate cumsum within each question_id
    #     def move_outlier_and_cumsum(df):
    #         outlier = df[df["cluster"] == -1]
    #         main = df[df["cluster"] != -1].sort_values("count", ascending=False)
    #         df_sorted = pd.concat([main, outlier], ignore_index=True)
    #         df_sorted["cum_prop"] = df_sorted["prop"].cumsum()
    #         return df_sorted

    #     cum_prop_cluster = cum_prop_cluster.groupby("question_id", group_keys=False).apply(move_outlier_and_cumsum)

    #     self.cum_prop_cluster = cum_prop_cluster

    #     print("Clusters generated; -1 indicates outliers. Empty insights remain with NaN clusters.")

    #     return self.cum_prop_cluster

    def clean_clusters(self, final_cluster_count: dict = None) -> pd.DataFrame:
        """
        Selects the top N clusters (by size) for each research question, marking all other clusters as outliers (-1).
        Updates self.state.insights with a new column 'selected_cluster' and saves the result.

        Args:
            final_cluster_count (dict): Dictionary mapping question_id to the number of clusters to keep for that question.
                                        Example: {'question_0': 3, 'question_1': 2, ...}

        Returns:
            pd.DataFrame: Updated insights DataFrame with 'selected_cluster' column.
        """
        if final_cluster_count is None:
            self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "08_clusters"))
            return(self.state.insights)

        else:
            rqs = self.state.insights["question_id"].unique()
            if len(rqs) != len(final_cluster_count):
                raise ValueError(
                    "final_cluster_count must specify the number of clusters to keep for each research question."
                )

            selected_clusters_list = []

            # Loop over each research question
            for rq in self.state.insights["question_id"].unique():
                # Filter insights for the current research question
                current_rq_df = self.state.insights[self.state.insights["question_id"] == rq].copy()
                # Count the size of each cluster (excluding outliers)
                cluster_sizes = current_rq_df.dropna(subset=["cluster"]).groupby("cluster").size().sort_values(ascending=False)

                # Get the number of clusters to keep for this question
                n_keep = final_cluster_count.get(rq, 0)
                # Get the cluster labels of the top N clusters (excluding outlier cluster -1)
                top_clusters = cluster_sizes[cluster_sizes.index != -1].head(n_keep).index.tolist()

                # Mark clusters to keep, others (and outliers) set to -1
                current_rq_df["selected_cluster"] = np.where(
                    current_rq_df["cluster"].isin(top_clusters),
                    current_rq_df["cluster"],
                    -1
                )

                selected_clusters_list.append(current_rq_df)

            # Concatenate all research questions back together
            self.state.insights = pd.concat(selected_clusters_list)
            # Save the updated DataFrame to disk
            self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "08_clusters"))
            return self.state.insights
        
            

class Summarize:
    def __init__(self,
                 state: Any,
                 llm_client: Any,
                 ai_model: str,
                 paper_output_length: int,  # Approximate total paper length in words
                 summary_save_location: str = None, 
                 pickle_save_location: str = config.PICKLE_SAVE_LOCATION,
                 state_save_location: str = os.path.join(config.STATE_SAVE_LOCATION, "12_summarize"), 
                 insight_embedding_path = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")):
        """
        Class to handle summarization of clustered insights.

        Args:
            state: Object holding insights (expects DataFrame `state.insights`).
            llm_client: Client to interact with LLM API.
            ai_model: Model name or identifier for LLM.
            paper_output_length: Total word length for paper; used to proportion cluster summaries.
            summaries_pickle_path: Optional path to pickle the resulting summaries DataFrame.
        """
        self.state: Any = deepcopy(state)
        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.paper_output_length: int = paper_output_length
        self.summary_save_location: str = config.SUMMARY_SAVE_LOCATION

        # Check that the embeddings have been created from the clustering step. If so, load. If not send the user back to run clustering
        if not os.path.exists(insight_embedding_path):
            raise FileNotFoundError(f"Insight embeddings pickle not found at {insight_embedding_path}. Please run clustering first or amend the path to where you pickled your insight embeddings.")
        else:
            with open(insight_embedding_path, "rb") as f:
                self.insight_embeddings_array: np.ndarray = pickle.load(f)

        # Reload any runs that might have already happened, or create place holders for that data. 
        # These are in a different form to state - as we start collapsing insights - so we want to be able to exhaust them sommewhere so all mutations can be tracked and inspected
        self.cluster_summary_list = self._reload_summary_outputs("cluster_summary") 
        self.theme_schema_list = self._reload_summary_outputs("theme_schema")
        self.mapped_theme_list = self._reload_summary_outputs("mapped_theme")
        self.populated_theme_list = self._reload_summary_outputs("populated_theme")
        self.redundancy = self._reload_summary_outputs("redundancy_pass")      
        
        # Ensure summary save location exists
        if not os.path.exists(self.summary_save_location):
            os.makedirs(self.summary_save_location, exist_ok=True)

    
    def _reload_summary_outputs(self, file_prefix: str) -> Optional[List[pd.DataFrame]]:
        """
        Finds and loads all parquet versions of a specific file, sorted by creation time.

        This method searches the summary save location for files matching the 
        pattern '{file_prefix}*.parquet' and returns them as a list of DataFrames 
        ordered from oldest to newest.

        Args:
            file_prefix: The base filename prefix to search for (e.g., 'cluster_summary').

        Returns:
            A list of pandas DataFrames if matching files exist, otherwise None.
            The list is sorted by file creation time (st_birthtime/st_mtime).
        """
        base_dir = Path(self.summary_save_location)
        
        # 1. Grab all potential matches using the prefix wildcard
        # We wrap in list() so we can check if any files were actually found
        paths = list(base_dir.glob(f"{file_prefix}*.parquet"))

        if not paths:
            return None

        # 2. Sort by creation/birth time
        # Uses st_birthtime (Creation) if available, falls back to st_mtime (Modified)
        paths_sorted = sorted(
            paths, 
            key=lambda p: getattr(p.stat(), 'st_birthtime', p.stat().st_mtime)
        )

        # 3. Load and return the list of DataFrames
        return [pd.read_parquet(p.absolute()) for p in paths_sorted]

    def _delete_summary_outputs(self, file_prefixes: List[str]) -> None:
        """
        Deletes summary output files matching the given prefixes.

        Args:
            file_prefixes: List of file prefixes to delete (e.g., ['cluster_summary']).
        """
        if not self.summary_save_location:
            return

        base_dir = Path(self.summary_save_location)
        
        # Ensure we don't try to glob an empty path or a non-existent directory
        if not base_dir.is_dir():
            return

        for prefix in file_prefixes:
            # glob finds every file starting with prefix and ending in .parquet
            for file_path in base_dir.glob(f"{prefix}*.parquet"):
                # missing_ok=True prevents crashes if another process 
                # deletes the file between globbing and unlinking
                file_path.unlink(missing_ok=True)

    def _calculate_summary_length(self) -> pd.DataFrame:
        """
        Calculate approximate word length for each cluster relative to the total paper

        Returns:
            DataFrame of insights with additional 'length_str' column for prompting the LLM.
        """
        # Count number of insights per cluster
        length_df: pd.DataFrame = (
            self.state.insights
            .dropna(subset = ["cluster"]) # remove any cases where chunks revealed no insights and therefore have no cluster
            .groupby(["question_id", "cluster"])
            .size()
            .reset_index(name="count")
        )

        # Compute proportion of total insights and allocate word length per cluster
        length_df["prop"] = length_df["count"] / length_df["count"].sum()
        length_df["length"] = length_df["prop"] * self.paper_output_length
        length_df["length_str"] = np.where(
            length_df["length"] > 2800,
            "2800 words (approx 4000 tokens)",
            length_df["length"].astype(int).astype(str) + " words"
        )

        # Merge length info back to original insights DataFrame
        insights_with_length: pd.DataFrame = self.state.insights.merge(
            length_df[["question_id", "cluster", "length_str"]],
            how="left",
            on=["question_id", "cluster"]
        )

        return insights_with_length

    def _calculate_centroid(self, col="full_insight_embedding"):
        rows = []
        for rq, d in self.state.insights.groupby("question_id", sort=False):
            # get clusters with at least one non-null embedding
            for cl, g in d.groupby("cluster", sort=False):
                vecs = [v for v in g[col].tolist() if isinstance(v, (list, tuple, np.ndarray))]
                if not vecs:
                    continue  # skip empty cluster
                A = np.asarray(vecs, dtype=np.float32)
                # guard ragged
                if not np.all([len(v) == A.shape[1] for v in A]):
                    # filter by the modal length
                    L = pd.Series([len(v) for v in vecs]).mode().iloc[0]
                    A = np.asarray([v for v in vecs if len(v) == L], dtype=np.float32)
                # drop rows with NaN
                mask = ~np.isnan(A).any(axis=1)
                A = A[mask]
                if A.size == 0:
                    continue
                centroid = A.mean(axis=0, dtype=np.float32)
                rows.append({"question_id": rq, "cluster": cl, "centroid": centroid})
        
        return pd.DataFrame(rows, columns=["question_id", "cluster", "centroid"])

    def _estimate_shortest_path(self):
        print("Calculating centroids for each cluster...")
        centroids = self._calculate_centroid(col="full_insight_embedding")

        # handle outliers later
        centroids = centroids[centroids["cluster"] != -1].copy()
        print("Estimating shortest path through clusters for each research question...")
        shortest_paths = {}

        for rq, df in centroids.groupby("question_id", sort=False):
            clusters = df["cluster"].tolist()
            C = np.stack(df["centroid"].to_list()).astype(np.float32)

            # prefer cosine distance on embeddings
            # normalize to unit vectors
            norms = np.linalg.norm(C, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            U = C / norms
            # cosine distance matrix
            D = 1.0 - (U @ U.T)
            np.fill_diagonal(D, 0.0)

            n = len(clusters)
            if n <= 1:
                order = clusters
            elif n < 10:
                best_len = np.inf
                best_perm = None
                for perm in itertools.permutations(range(n)):
                    length = sum(D[perm[i], perm[i+1]] for i in range(n-1))
                    if length < best_len:
                        best_len = length
                        best_perm = perm
                order = [clusters[i] for i in best_perm]
                final_order = order + [-1]  # add outlier cluster at the end
            else:
                G = nx.Graph()
                for i in range(n):
                    for j in range(i+1, n):
                        G.add_edge(clusters[i], clusters[j], weight=float(D[i, j]))
                order = nx.approximation.traveling_salesman_problem(G, weight="weight", cycle=False)
                final_order = order + [-1]  # add outlier cluster at the end
            shortest_paths[rq] = {"order": final_order}
    
        return shortest_paths
    

    def summarize_clusters(self) -> Summaries:
        """
        Generate summaries for all clusters across all research questions.

        Returns:
            Summaries object containing a DataFrame of cluster summaries.
        """
        
        if self.cluster_summary_list is not None and len(self.cluster_summary_list) > 0:
            new = None
            while new not in ["1", "2"]:
                new = input(
                    "Cluster summaries already exist on disk. Do you want to:\n"
                    "(1) reload existing summaries\n"
                    "(2) regenerate summaries (NOTE:this will overwrite existing summaries and delete all existing theme mapping, populated themes, orphans and redundancy passes; as these all derive from summaries)? \n"
                    "Enter 1 or 2:\n"
                ).lower()
            if new == "1":
                print(
                    "Summaries loaded.\n" \
                    "Summaries can be accessed via the cluster_summary_list attribute of this class. There is only one item in the list thus: `variable.cluster_summary_list[0]`."
                    )
                return(None) # Return to exit the function and avoid re-running summarization
            else:
                # If we are regenerating summeries we need to delete all existing outputs and reset the attributes to none as they are loaed on init if they exist
                print("Re-running summarization of clusters...")
                self._delete_summary_outputs(file_prefixes=["cluster_summary", "theme_schema", "populated_theme", "orphan", "redundancy_pass"])
                self.cluster_summary_list = None
                self.theme_schema_list = None
                self.populated_theme_list = None
                self.orphans = None
                self.redundancy = None

        # We are going to send the insights to the LLM in the order of the shortest path, so that the most similar clusters are summarized close together
        # This will add coherence to the final paper when the summaries are stitched together
        # It will also aid in the applicaion of the sliding window for summary clean up
        shortest_paths = self._estimate_shortest_path()
        
        # Add calculated lengths to insights
        self.state.insights = self._calculate_summary_length()
        
        # Create list to populate with summaries from the LLM
        summaries_dict_lst: List[dict] = []

        # Get the numbers to show progress 
        total_clusters = len(self.state.insights.groupby("question_id")["cluster"].nunique(dropna=False))
        count = 1

        # Loop over unique research questions
        for _, row in self.state.questions.iterrows():
            rq_id = row["question_id"]
            rq_text = row["question_text"]
            rq_df: pd.DataFrame = self.state.insights[self.state.insights["question_id"] == rq_id].copy()

            # Loop over clusters for this research question - in shortest path order
            for cluster in shortest_paths[rq_id]["order"]:
                print(f"Summarizing cluster {cluster} for research question {rq_id} ({count} of {total_clusters})...")
                count += 1
                # Skip any cases where chunks might have had no insights (and therefore no cluster)
                if pd.isna(cluster) or cluster == "NA":
                    continue

                cluster_df: pd.DataFrame = rq_df[rq_df["cluster"] == cluster]
                length_str: str = cluster_df["length_str"].iloc[0]
                # get the insights, they are list of single strings. So make sure they are valid string to send to the LLM
                insights: List[str] = cluster_df["insight"].apply(
                    lambda x: x if isinstance(x, str) else (
                        x[0] if isinstance(x, list) and len(x) == 1 and isinstance(x[0], str) else None
                    )).tolist()
                
                if any(i is None for i in insights):
                    raise ValueError("Insight format error: each insight must be a string or a single-item list containing a string.")

                # Build system prompt from predefined method
                sys_prompt: str = Prompts().summarize(summary_length=length_str)

                # Get the summaries frozen so far if there are any
                frozen_summaries = pd.DataFrame(summaries_dict_lst)["summary"].tolist() if summaries_dict_lst else []

                # Build user prompt for LLM
                user_prompt: str = (
                    f"Research question id: {rq_id}\n"
                    f"Research question text: {rq_text}\n"
                    "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
                    f"{'\n'.join(frozen_summaries) if frozen_summaries else ''}\n"
                    f"Cluster: {cluster}\n"
                    "INSIGHTS:\n" +
                    "\n".join(insights)
                )

                json_schema = {
                    "name": "cluster_summary",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "question_text": {"type": "string"},
                            "cluster": {"type": "number"},
                            "summary": {"type": "string"}
                        },
                        "required": ["question_id", "question_text", "cluster", "summary"],
                        "additionalProperties": False
                    }
                }
                fall_back = {"question_id": rq_id, "question_text": rq_text, "cluster": cluster, "summary": ""}

                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    json_schema=json_schema,
                    fall_back=fall_back
                )

                summaries_dict_lst.append(response)

        # Now convert the list of dict responses to a dataframe for easier handling and saving
        summaries_df: pd.DataFrame = pd.DataFrame(summaries_dict_lst)

        print(
            f"Summaries saved here: {self.summary_save_location} and accesible via `variable.cluster_summary_list[0]`.\n"
        )

        self.cluster_summary_list = [summaries_df]

        # Save summaries as this is LLM output we may want to reuse later - save as parquet
        os.makedirs(self.summary_save_location, exist_ok=True)
        [i.to_parquet(os.path.join(self.summary_save_location, "cluster_summary.parquet")) for i in self.cluster_summary_list]

        return self.cluster_summary_list
    

    def gen_theme_schema(self, save_file_name="theme_schema.parquet") -> pd.DataFrame:
        # First check that summaries exist
        if self.cluster_summary_list is None or len(self.cluster_summary_list) == 0:
            raise ValueError("No summaries found. Please run summarize() first.")

        # Next we want to confirm that the user wants to run the mapping again, rather tha possibly accessing previoulsy generated and saved mappings
        # There are actually three options here: 1) start the mapping process afresh, 2) load previously generated maps, 3) run another mapping iteration on top of a populated theme. We have to handle all
        # First we check if there are theme maps (which get loaded on class init if they exist)
        new = None
        if self.theme_map_list is not None and len(self.theme_map_list) > 0:
            while new not in ['1', '2', '3']:
                new = input("Theme mapping already exists. " \
                "Theme maps already exist. Please select what you want to do:\n" \
                "1: Load existing maps\n" \
                "2: Fully recreate your theme mapping (i.e. start again). NOTE this will delete all existing theme maps as well as any populated themes, orphans and redundancy passes\n" \
                "3: Run another mapping iteration on top of a populated theme\n" \
                "Enter either 1, 2, or 3:\n").lower()
            
            if new == '1':
                print("Theme mapping available via .theme_map_list, orderd by creation time if multiple exist.")
                return(None)
            
            elif new == '2':
                print("Re-running theme mapping from scratch...")
                self._delete_summary_outputs(file_prefixes=["theme_schema", "populated_theme", "orphan", "redundancy_pass"])
                self.theme_schema_list = None
                self.populated_theme_list = None
                self.orphans = None
                self.redundancy = None

            else:
                # Make sure the user is not iterating theme mapping without first populating themes for whatever iteration of this they are on.
                if len(self.theme_schema_list) > len(self.populated_theme_list):
                    raise ValueError("Your theme schema is already ahead of your theme population. Please run populate_themes() before generating theme schema again")
                # Make sure they have populated themes before they try to iterate the theme schema
                if len(self.populated_theme_list) == 0:
                    raise ValueError("No populated themes found. Please run populate_themes() before tying to iterate your theme schema.")
                
                # Run new mapping
                print("Running additional theme mapping...")

        # Now we create a generic dataframe from eithe the summaries or the populated themes depending on whether the user is starting fresh or iterating
        # First check new or full re-write. 
        if new == '1' or self.theme_schema_list is None or new == '2':
            # Grab data from summaries
            source_df = self.summaries_list[0]
            source_df.rename(columns={"cluster": "id", "summary": "text_to_theme"}, inplace=True)
        else:
            # if its an iteration we get data from the last populated theme and convert the columsn to a generic form to send to the llm
            source_df = self.populated_theme_list[-1]
            source_df.rename(columns={"theme_id": "id", "theme_text": "text_to_theme"}, inplace=True) ##########################DOUBLE CHECK THIS RENAME TO MAKE SURE IT FITS WITH THE EXPECTED PROMPT AND LLM RESPONSE
        
        out_df_list = []

        for _, row in self.state.questions.iterrows():
            question_id = row["question_id"]
            question_text = row["question_text"]
            rq_df = source_df[source_df["question_id"] == question_id].copy()
            text_to_theme = "\n\n".join(rq_df["text_to_theme"].tolist())

            user_prompt = (
                f"Research Question: {question_text}\n"
                "TEXT TO ANALYZE:\n"
                f"{text_to_theme}\n"
            )
            sys_prompt = Prompts().gen_theme_schema()

            fall_back = {"question_id": question_id, "themes": []}

            json_schema = {
                "name": "thematic_schema_generator",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                    "themes": {
                        "type": "array",
                        "items": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string" },
                            "label": { "type": "string" },
                            "instructions": { "type": "string" }
                        },
                        "required": ["id", "label", "instructions"],
                        "additionalProperties": false
                        }
                        }
                    },
                    "required": ["themes"],
                    "additionalProperties": false
                    }
                }

            response = utils.call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                fall_back=fall_back,
                return_json=True,
                json_schema=json_schema
            )

            theme_list = response.get("themes", [])
            themes_df = pd.DataFrame(theme_list, columns=["id", "label", "criteria"])
            themes_df["question_id"] = question_id
            themes_df["question_text"] = question_text

            out_df_list.append(themes_df)
        
        # Concat all the questions
        output = pd.concat(out_df_list, ignore_index=True, sort=False)
        # Append to the list of schemas
        self.theme_schema_list.append(output)
        # Save all the schemas in the list to capture the new one and maintain the order on disk in case we need to reload later for any reason
        os.makedirs(save_dir, exist_ok=True)
        for idx, df in enumerate(self.theme_schema_list):
            df.to_parquet(os.path.join(self.summary_save_location, f"theme_schema_{idx+1}.parquet"))

        return self.theme_schema_list[-1]

    
    def map_insights_to_themes(self, batch_size=75) -> pd.DataFrame:
        #### RESUMPTION/RECOVER/REPOPULATE LOGIC #### ---------------------------

        # First check whether a partial mapping took place - this is a process with a large number of calls so we are going to log to pickle as we go so that resume is possible
        # Create a default variable that can be populated if partial run exists
        mapped_insights_df_list = None
        
        # Now check for pickle file
        if os.path.exists(os.path.join(self.summary_save_location, "mapped_theme_in_progress.pickle")):
            resume = None
            while resume not in ["1", "2"]:
                resume = input("A partial mapping process was detected. Do you want to:\n"
                               "1) resume from the last saved point? \n"
                               "2) start a new mapping process? \n"
                               "Enter 1 or 2:\n").lower()
            if resume == "1":
                with open(os.path.join(self.summary_save_location, "mapped_theme_in_progress.pickle"), "rb") as f:
                    mapped_insights_df_list = pickle.load(f)
                    mapped_insights = pd.concat(mapped_insights_df_list, ignore_index=True)["insight_id"].tolist()
                    
                print("Resuming mapping process from last saved point...")

            else:
                print("Starting new mapping process and deleting the in progress pickle...")
                os.remove(os.path.join(self.summary_save_location, "mapped_theme_in_progress.pickle"))

        # Then check that theme schema exists
        if self.theme_schema_list is None or len(self.theme_schema_list) == 0:
            raise ValueError("No theme schema found. Please run gen_theme_schema() first.")
        
        # Then check that theme mapping is not ahead of theme population (as you don't want to map multiple times without populating)
        if len(self.theme_schema_list) > len(self.populated_theme_list):
            raise ValueError("Your theme schema is already ahead of your theme population. Please run populate_themes() before mapping insights to themes.")
        
        # Now check whether the user just want to re-load thier existing mappings of re-run the mapping on the latest schema
        if self.mapped_theme_list is not None and len(self.mapped_theme_list) > 0:
            new = None
            while new not in ["1", "2"]:
                new = input(
                    "Mapped themes already exist on disk. Do you want to:\n"
                    "(1) reload existing mapped themes\n"
                    "(2) remap insights to the current themes again (NOTE:this will overwrite existing mapped themes)? \n"
                    "Enter 1 or 2:\n"
                ).lower()
            if new == "1":
                print(
                    "Mapped themes loaded. Inspect them via variable.mapped_theme_list[-1]\n"
                )
                return(None) # Return to exit the function and avoid re-running the mapping
            else:
                # If re-running make sure that the number of theme maps is equal to the one less than the number of schemas (so we are mapping to the last one)
                schema_len = len(self.theme_schema_list)
                self.mapped_theme_list = self.mapped_theme_list[:schema_len - 1]             

        #### MAPPING LOGIC ####----------------------------
        # create the output for the loop using the recovered list if it exists, otherwise create an empty list to populate with the mapping results as we go
        mapped_insights_df_list = [] if mapped_insights_df_list is None else mapped_insights_df_list
        # Create a temporary copy of the insights to work with so that we can drop mapped insights as we go without affecting the original state
        temp_state_insights = self.state.insights.copy()
        # If we are resuming we need to drop the already mapped insights
        temp_state_insights = temp_state_insights[~temp_state_insights["insight_id"].isin(mapped_insights)] if mapped_insights_df_list is not None else temp_state_insights
        
        # Iterate through each research question
        for _, q_row in self.state.questions.iterrows():
            question_id = q_row["question_id"]
            question_text = q_row["question_text"]

            # Filter insights for this question and drop empty/NaN values immediately
            q_insights_df = temp_state_insights[
                (temp_state_insights["question_id"] == question_id) & 
                (temp_state_insights["insight"].notna()) & 
                (temp_state_insights["insight"] != "")
            ].copy()

            # Get the schema for this specific question
            q_schema_df = self.theme_schema_list[-1][
                self.theme_schema_list[-1]["question_id"] == question_id
            ].copy()
            q_schema_json = q_schema_df[["id", "label", "criteria"]].to_json(orient="records", indent=2)

            # create json schema for the llm call so i don't recreate it every call
            json_schema = {
                "name": "insight_to_theme_mapper",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "mapped_data": { # Match prompt: "mapped_data"
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "insight_id": {"type": "string"},
                                    "theme_ids": { # Match prompt: "theme_ids"
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["insight_id", "theme_ids"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["mapped_data"],
                    "additionalProperties": False
                }
            }
            
            # CHUNKING LOGIC: Process the filtered insights in batches
            for i in range(0, len(q_insights_df), batch_size):
                batch_df = q_insights_df.iloc[i : i + batch_size]
                
                # Format insights for the prompt: "id: text"
                current_batch_str = "\n".join(
                    [f"{row.insight_id}: {row.insight}" for row in batch_df.itertuples()]
                )

                sys_prompt = self.theme_map_to_schema() # Assuming this is your prompt method
                user_prompt = (
                    f"RESEARCH QUESTION: {question_text}\n"
                    "THEMATIC CODEBOOK:\n"
                    f"{q_schema_json}\n\n"
                    f"INSIGHTS TO MAP:\n"
                    f"{current_batch_str}\n\n"
                )

            response = utils.call_chat_completion(
                sys_prompt=sys_prompt,  
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                fall_back={"mappings": []}, # Match your schema key
                return_json=True, 
                json_schema=json_schema
            )

            # Convert response to DF and tag with metadata
            batch_results_df = pd.DataFrame(response.get("mapped_data", []))

            if not batch_results_df.empty:
                # 1. Assignment is required (explode is not in-place)
                # 2. Use the correct key "theme_ids"
                batch_results_df = batch_results_df.explode("theme_ids")
                
                batch_results_df["question_id"] = question_id
                mapped_insights_df_list.append(batch_results_df)

                with open(os.path.join(self.summary_save_location, "mapped_theme_in_progress.pickle"), "wb") as f:
                    pickle.dump(mapped_insights_df_list, f)

        # Concat everything
        if not mapped_insights_df_list:
            mapped_insights_df = pd.DataFrame()
        else:
            mapped_insights_df = pd.concat(mapped_insights_df_list, ignore_index=True)

        self.mapped_theme_list.append(mapped_insights_df)
        os.makedirs(self.summary_save_location, exist_ok=True)
        # delete the in progress pickle as we have now saved the final output to the mapped theme list and we don't want to accidentally resume from a mid mapping point when we have a full mapping saved
        if os.path.exists(os.path.join(self.summary_save_location, "mapped_theme_in_progress.pickle")):
            os.remove(os.path.join(self.summary_save_location, "mapped_theme_in_progress.pickle"))

        for idx, i in enumerate(self.mapped_theme_list):
            i.to_parquet(os.path.join(self.summary_save_location, f"mapped_themes_{idx+1}.parquet"), index=False)

        return(self.mapped_theme_list[-1])


    def populate_themes(self, save_file_name="populated_themes.parquet") -> pd.DataFrame:
        # utils
        def build_frozen_block(frozen_content: list[dict]) -> str:
            if not frozen_content:
                return "(none)\n"
            parts = []
            for t in frozen_content:
                parts.append(
                    f"Theme ID: {t.get('theme_id','')}\n"
                    f"Label: {t.get('label','')}\n"
                    f"Criteria: {t.get('criteria','')}\n"
                    f"Content:\n{t.get('contents','')}\n"
                    "--- END THEME ---"
                )
            return "\n".join(parts) + "\n"

        def build_remaining_themes_block(rq_df: pd.DataFrame, current_theme_id: str, processed_ids: set[str]) -> str:
            # remaining = all themes for this RQ not yet processed, excluding the current one
            rem = rq_df.loc[~rq_df["id"].isin(processed_ids | {current_theme_id}), ["label", "criteria"]]
            if rem.empty:
                return "(none)\n"
            parts = []
            for _, r in rem.iterrows():
                parts.append(
                    f"Theme label: {r['label']}\n"
                    f"Criteria: {r['criteria']}\n"
                    "--- END THEME ---"
                )
            return "\n".join(parts) + "\n"

        # guard
        if not hasattr(self, "summary_themes"):
            raise ValueError("No summary themes found. Please run identify_themes() first.")

        save_dir = self.summary_save_location
        save_path = os.path.join(save_dir, save_file_name)

        if os.path.exists(save_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input("Populated themes already exist on disk. Recover (r) or generate new (n)? ").lower()
            if recover == 'r':
                self.populated_themes = pd.read_parquet(save_path)
                return self.populated_themes
            else:
                print("Re-running population of themes...")

        out_rows = []
        total_themes = len(self.summary_themes)
        counter = 0

        # iterate per research question
        for question_id, rq_df in self.summary_themes.groupby("question_id", sort=False):
            # reset frozen content per question to avoid leakage
            frozen_content: list[dict] = []
            processed_ids: set[str] = set()

            # source text for this RQ
            summary_text_list = self.summaries.loc[self.summaries["question_id"] == question_id, "summary"].tolist()
            summary_text = "\n\n".join(summary_text_list)

            # iterate themes for this question in the given order
            for _, row in rq_df.iterrows():
                counter += 1
                print(f"Populating theme {counter} of {total_themes}")

                question_text = row["question_text"]
                theme_id = row["id"]
                theme_label = row["label"]
                theme_criteria = row["criteria"]

                frozen_block = build_frozen_block(frozen_content)
                remaining_theme_block = build_remaining_themes_block(rq_df, theme_id, processed_ids)

                sys_prompt = Prompts().populate_themes()
                user_prompt = (
                    f"Research question id: {question_id}\n"
                    f"Research question text: {question_text}\n"
                    "FROZEN CONTENT (read-only; text already assigned to themes):\n"
                    f"{frozen_block}"
                    "---CURRENT THEME TO POPULATE:---\n"
                    f"Theme ID: {theme_id}\n"
                    f"Theme label: {theme_label}\n"
                    f"Criteria: {theme_criteria}\n\n"
                    "CLUSTER SUMMARY TEXT (source material):\n"
                    f"{summary_text}\n\n"
                    "--- THEMES STILL TO PROCESS (context only):---\n"
                    f"{remaining_theme_block}"
                )

                fall_back = {"question_id": question_id, "theme_id": theme_id, "assigned_content": ""}

                resp = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    fall_back=fall_back,
                )

                assigned = (resp.get("assigned_content") or "").strip()

                out_row = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "theme_id": theme_id,
                    "label": theme_label,
                    "criteria": theme_criteria,
                    "contents": assigned,
                }
                out_rows.append(out_row)

                # update frozen and processed sets
                frozen_content.append(out_row)
                processed_ids.add(theme_id)

        output = pd.DataFrame(
            out_rows, columns=["question_id", "question_text", "theme_id", "label", "criteria", "contents"]
        )

        self.populated_themes = output
        os.makedirs(save_dir, exist_ok=True)
        self.populated_themes.to_parquet(save_path)
        return self.populated_themes
        