
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
        state: QuestionState = None,
        papers: pd.DataFrame = None,
        confirm_read: Optional[str] = None,
        file_path: str = os.path.join(os.getcwd(), "data", "docs"),
    ) -> None:
        """Initialize Ingestor and validate state/papers format."""
        self.state = deepcopy(
            utils.validate_format(
                state=state,
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
        self.confirm_read: Optional[str] = confirm_read # Param controls whether user has confirmed reading ingestion errors
        self.ingestion_errors: List[str] = []

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
        return list_of_files

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
        
        working_insights = self.state.insights.copy()
        list_of_papers_by_page: List[List[str]] = []
        ingestion_status: List[int] = [] # to track ingestion success (1) or failure (0)
        self.ingestion_errors = []

        list_of_files = self._list_files()
        
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
            self.confirm_read = self.confirm_read or ""
            while self.confirm_read != "c":
                self.confirm_read = input(
                    "Ingestion errors occurred. Examine .ingestion_errors and state.full_text.\n"
                    "Hit 'c' to confirm having read this message:\n"
                ).lower()

        # Create an ingestion status dataframe tracking ingestion success/failure linked by paper_id and question_id
        ingestion_status_df = pd.DataFrame({
            "paper_id": [os.path.splitext(os.path.basename(path))[0] for path in list_of_files],
            "question_id": [os.path.basename(os.path.dirname(path)) for path in list_of_files],
            "ingestion_status": ingestion_status
        })
        
        # Get all the file imports that matched an id
        matched_ids_df = working_insights.merge(
            ingestion_status_df, how="left", on=["question_id", "paper_id"]
        ) \
        .assign(ingestion_ids_matched = lambda x: np.where(x["ingestion_status"].isna(), 0, 1)) # On a left join if there is no id match ingestion status will be na

        # Get all the ids for which no file was found
        failed_id_matches = matched_ids_df[matched_ids_df["ingestion_ids_matched"] == 0] # Above we set this to 0 or 1, so 0 means no match found
        self.failed_id_matches= failed_id_matches

        if failed_id_matches.shape[0] > 0:
            abort = None
            while abort not in ['y', 'n']:
                abort = input(
                f"Warning: {failed_id_matches.shape[0]} paper(s) in the insights table "
                "did not have a matching file in the ingestion directory. This is not neccessarily an error, but if you want to be able to match " 
                "these papers to thier search terms and search engines later you will need to ensure the files are named correctly.\n\n"
                "If you are conducting a literature review this warning is likely relevant to you. If you are reading your own corpus, you can likely ignore this message.\n\n"
                "If you ignore this warning any paper ids that did not have a matching file will be deleted from state.insights. "  
                "You can look these up later by exploring the state.insights object created earlier in the pipeline.\n\n"
                "Do you wish to abort ingestion to review the failed id matches? (y/n):\n"
            ).lower()
            
            if abort == 'y':
                print("Aborting ingestion. Please review the failed id matches (returned below and accessible via .failed_id_matches).")
                return(failed_id_matches)
            else:
                pass #continue with ingestion despite failed id matches

        # Now get all the file imports that did not match an id
        unmatched_id_df = (
            working_insights.merge(ingestion_status_df, how="outer", on=["question_id", "paper_id"], indicator=True) 
            .query('_merge == "right_only"')  # identify rows in ingestion_status_df not matched in working_insights - i.e. they came from the right side of the join only 
            .assign(ingestion_ids_matched = 0)
            .drop(columns=["_merge"])
            .assign(paper_id = lambda x:[f"unmatched_paper_{i+1}" for i in range(x.shape[0])])
        ) 

        # Dropping the na values from matched_ids_df that showed failed id matches
        # This means we effectively lose track of those papers that the retrival module might have identified but the user did not download. 
        # While this is a loss of data it reflects an intentional choice - ReadingMachine reads the corpus of texts that you put in the import folder, it does not guarantee tracking of your files that you failed to get into this folder. 
        # The retrieval moduel helps the reader populate that folder, but the user's capacity to populate it lies outside of ReadingMachines core concern. 
        matched_ids_df = matched_ids_df[matched_ids_df["ingestion_status"].notna()]

        # Combine matched and unmatched dataframes into final insights
        working_insights = pd.concat([matched_ids_df, unmatched_id_df], ignore_index=True)
        working_insights.drop(columns=["ingestion_ids_matched"], inplace=True)
        self.state.insights = working_insights
        
        # Build full_text DataFrame
        full_text = pd.DataFrame({
            "paper_path": list_of_files,
            "pages": list_of_papers_by_page
        })
        full_text["paper_id"] = [os.path.splitext(os.path.basename(path))[0] for path in list_of_files]
        full_text["question_id"] = [os.path.basename(os.path.dirname(path)) for path in list_of_files]
        full_text["full_text"] = ["".join(pages) for pages in full_text["pages"]]

        # Drop pages from full_text as they take up memory and are not needed:
        full_text.drop(columns=["pages"], inplace=True) 

        self.state.full_text = full_text
        return self.state.full_text

    def _get_metadata(self, paper_id: str, text: str) -> dict[str, Any]:
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

    def update_metadata(self) -> pd.DataFrame:
        """Update metadata for papers missing it by calling the LLM."""
        
        metadata_check_df = self.state.insights.copy()

        metadata_check_df = metadata_check_df[
            ["paper_id", "paper_title", "paper_author", "paper_date"]
        ].merge(
            self.state.full_text[["paper_id", "full_text"]],
            how="left",
            on=["paper_id"]
        )

        # Metadata is a first order concern here and it can get mixed up if the user misclasifies filenames on import, so this approach double checks all metadata for all papers, even those that has imports currently.
        # This is slightly expensive but its more problematic if insights get attributed to the wrong authors. This approach is the only way to ensure every paper has the correct metadata. 
        for idx, row in metadata_check_df.iterrows():
            print(f"Checking metadata for paper {idx + 1} of {self.state.insights.shape[0]}...")
            paper_id = row["paper_id"]
            text = row["full_text"][:5000] if row["full_text"] else ""
            metadata = self._get_metadata(paper_id, text)
            metadata_check_df.at[idx, "paper_title"] = metadata["paper_title"]
            metadata_check_df.at[idx, "paper_author"] = metadata["paper_author"]  
            metadata_check_df.at[idx, "paper_date"] = metadata["paper_date"]

        # drop full_text from insights as it is not needed and takes up memory - we only needed it for the metadata check, now that is done we can drop it to keep the insights dataframe clean and efficient
        metadata_check_df = metadata_check_df.drop(columns=["full_text"])
        
        # Update state.insights with the new metadata
        self.state.insights = metadata_check_df

        self.state.save(config.STATE_SAVE_LOCATION)

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
        self.state.full_text["chunks"] = chunks_list
        self.state.chunks = self.state.full_text[["question_id", "paper_id", "chunks"]].explode("chunks").reset_index(drop=True).copy()
        self.state.chunks.rename(columns={"chunks": "chunk_text"}, inplace=True)   
        self.state.chunks["chunk_id"] = self.state.chunks.groupby(["paper_id"]).cumcount()

        # Chunks from full_text as its now joined by paper and question id
        self.state.full_text.drop(columns=["chunks"], inplace=True)

        # Save the updated state
        self.state.save(config.STATE_SAVE_LOCATION)

class Insights:
    def __init__(
        self,
        state: "QuestionState",
        llm_client: Any,
        ai_model: str, 
        pickle_path: str = os.path.join(os.getcwd(), "data", "pickles"), 
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

        self.chunk_insights_pickle_file = chunk_insights_pickle_file
        self.meta_insights_pickle_file = meta_insights_pickle_file


        # Ensure state has all required columns before processing
        self.state = deepcopy(
            utils.validate_format(
            state=state,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                 "download_status", "messy_question_id", "messy_paper_id","ingestion_status"
            ],
            injected_required_cols=None
            )
        )

        self.state.enforce_canonical_question_text()

    

    def _generate_chunk_insights(self, chunk_state= None, insights = None, count_start = None) -> pd.DataFrame:
        """
        Extract insights from each text chunk using the LLM.
        Each chunk is processed individually, and assessed relative to all the research questions.
        Insights are traced back to 
        (chunk_id, paper_id).

        Returns:
            pd.DataFrame: Updated `state.insights` with new insights appended.
        """
        if chunk_state is None:
            chunk_state = self.state.chunks
        if insights is None:
            insights: List[Dict[str, Any]] = []
        if count_start is None:
            count_start = 0
        
        # Generate a list of all the research questions with ids in the form <rq_id>: <rq_text> for the llm to consdier against each chunk
        rqs_df = self.state.insights[["question_id", "question_text"]].dropna().drop_duplicates()
        rqs_ids = [f"{row['question_id']}: {row['question_text']}" for _, row in rqs_df.iterrows()]
        rqs_ids_str = "\n".join(rqs_ids)


        # Merge chunk text with metadata (author, date, etc.)
        temp_state_df: pd.DataFrame = chunk_state.merge(
            self.state.insights[["paper_id", "question_text", "paper_author", "paper_date"]],
            how="left",
            on=["paper_id"]
        )

        # Iterate over each chunk
        chunk_insights_lst = []
        for idx, (df_index, row) in enumerate(temp_state_df.iterrows()):
            print(f"Processing chunk {idx + 1 + count_start} of {temp_state_df.shape[0]}...")

            # Extract fields from row
            paper_id: str = row["paper_id"]
            chunk_text = row["chunk_text"] if pd.notna(row["chunk_text"]) else ""
            chunk_id: int = int(row["chunk_id"])

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
           

            # Encode text safely for JSON
            safe_research_qs: str = json.dumps(rqs_ids_str, ensure_ascii=False)
            safe_chunk_text: str = json.dumps(chunk_text, ensure_ascii=False)
            safe_citation: str = json.dumps(citation, ensure_ascii=False)


            # Build prompts
            sys_prompt: str = Prompts().gen_chunk_insights()
            user_prompt: str = (
                f"RESEARCH QUESTIONS:\n{safe_research_qs}\n\n"
                f"TEXT CHUNK:\n{safe_chunk_text} - {safe_citation}\n"
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
            response_df = (pd.DataFrame(response_dict['results'].items(), columns=["question_id", "insight"])
                           .explode("insight") #explode on insights to make tidy data from the lists
                           .reset_index(drop=True))
            
            # Add paper_id and chunk_id to the response_df 
            response_df["chunk_id"] = chunk_id
            response_df["paper_id"] = paper_id

            # Append to insights list
            chunk_insights_lst.append(response_df)
            with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "wb") as f:
                pickle.dump(chunk_insights_lst, f)

        # Convert insights list to DataFrame
        print("Converting insights to DataFrame and merging into state...")
        chunk_insights_df: pd.DataFrame = pd.concat(chunk_insights_lst).reset_index(drop=True)
        print(f"Dropping cols for first merge...")
        
        # Merge into global insights table - first drop existing insight columns if present (these can be created by previous runs of recover_chunk_insights_generation)
        for col in ["chunk_id", "insight"]:
            if col in self.state.insights.columns:
                self.state.insights.drop(columns=[col], inplace=True)
        
        # Now we merge this chunk insights with all the insights data and metadata. 
        # Notably we drop question_id from the state.insights as previously if papers wwere not associated with a question when importing them, the question_id was NA
        # Now we have all the chunks and thier insights associated with a question_id thus this becomes the primary df
        working_insights_df = (
            chunk_insights_df[["question_id", "paper_id", "chunk_id", "insight"]]
            .merge(
                self.state.insights.drop(columns=["question_id"]),
                how="left",
                on=["paper_id", "chunk_id"]
                ))

        self.state.insights = working_insights_df

        # Note i don't save here as i only save at the end of the class's operations. This is cleaner for the user. 
        # Also since i have the recover function in place, once the insights are generated for this class it should be quick to recreate to this point
        return self.state.insights
    
    
    def _recover_chunk_insights_generation(self):
        print("Opening pickle file to recover chunk insights generation...")
        with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "rb") as f:
            recover_chunk_insights = pickle.load(f)
        
        start = len(recover_chunk_insights)
        print(f"Resuming chunk insights generation from chunk {start}...")
        self.generate_chunk_insights(chunk_state = self.state.chunks.iloc[start:], insights=recover_chunk_insights, count_start=start)


    def get_chunk_insights(self, chunk_state= None, insights = None, count_start = None) -> pd.DataFrame:
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
    
    def get_meta_insights(self, max_token_length = 100000) -> pd.DataFrame:
        """
        Generate 'meta-insights' — arguments that span multiple chunks within 
        the same paper. Each paper is processed once, combining all chunk insights 
        and the full text.

        Returns:
            pd.DataFrame: DataFrame of meta-insights appended to state.insights.

        Raises:
            ValueError: If chunk insights do not exist prior to running.
        """
        # Must run chunk insights first
        if "insight" not in self.state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        meta_insights: List[Dict[str, Any]] = []
        # All research questions for context
        rqs = self.state.insights[["question_id", "question_text"]].drop_duplicates().dropna()

        # rqs: List[str] = [
        #     f"{row['question_id']}: {row['question_text']}"
        #     for _, row in self.state.insights[["question_id", "question_text"]].iterrows()
        # ]

        # Process each paper
        for idx, paper_id in enumerate(self.state.insights["paper_id"].unique()):
            print(f"Processing meta-insight for paper {idx + 1} of {len(self.state.insights['paper_id'].unique())}...")
            # Get paper full text
            paper_content: str = (
                self.state.full_text
                .loc[self.state.full_text["paper_id"] == paper_id, "full_text"]
                .iloc[0]
            )

            # Check that the whole paper fits in the model context window, if not break into chunks and process separately 
            # (this is a bit of a hack but it allows us to at least get some meta insights from papers that exceed the context window, 
            # which is likely to be the case for many academic papers with the full text included)
            token_count = self.estimate_tokens(paper_content, self.ai_model)
            if token_count > max_token_length:
                paper_content_list = self.string_breaker(paper_content, max_token_length=max_token_length)
            else:
                paper_content_list = [paper_content]
              
            for paper_content in paper_content_list:
                # Get the insights dataframe for the specific paper_id 
                paper_df: pd.DataFrame = self.state.insights[self.state.insights["paper_id"] == paper_id]

                # Now get the metadata and chunk insights for the paper
                authors = paper_df['paper_author'].iloc[0]
                if isinstance(authors, list):
                    author_str = ", ".join(authors)
                elif pd.isna(authors):
                    author_str = ""
                else:
                    author_str = str(authors)
                date = paper_df['paper_date'].iloc[0]
                date_str = "" if pd.isna(date) else str(date)
                title = paper_df['paper_title'].iloc[0]
                title_str = "" if pd.isna(title) else str(title)
                metadata = f"{author_str}, {date_str}, {title_str}"

                # Any paper_id can be associated with multiple research questions. 
                # The link between papers and research questions is set when we determine if chunks for the paper hold any insighs for any RQ.
                # Thus a paper_id is associated with all the RQs for which any of its chunks hold insights. 
                # So we get all the RQs for the paper ID
                relevant_rqs_dict = paper_df[paper_df["insight"].notna()]["question_id", "question_text"].drop_duplicates().to_dict(orient="records")
                # Now iterate over each question for which the paper has relevance and call the llm to id the meta insights for that paper
                for rq_id, rq_text in zip(relevant_rqs_dict["question_id"], relevant_rqs_dict["question_text"]):
                    current_rq_str: str = f"{rq_id}: {rq_text}"
                    other_rqs_df: pd.DataFrame = rqs[rqs["question_id"] != rq_id] # Get other RQs for context, excluding the current one
                    other_rqs_str: str = "\n".join(f"{row['question_id']}: {row['question_text']}" for _, row in other_rqs_df.iterrows())
                    insights_str: str = "\n".join(paper_df.loc[paper_df["question_id"] == rq_id, "insight"].dropna().tolist()) # Get insights for the current RQ and paper ID, dropna to avoid issues with empty insights


                    # Build prompt
                    user_prompt: str = (
                        "SPECIFIC RESEARCH QUESTION FOR CONSIDERATION\n"
                        f"{current_rq_str}\n"
                        f"PAPER ID: {paper_id}:\n"
                        f"PAPER METADATA:\n"
                        f"{metadata}\n"
                        "PAPER TEXT:\n"
                        f"{paper_content}\n"
                        "EXISTING CHUNK INSIGHTS\n"
                        f"{insights_str}\n"
                        "OTHER RESEARCH QUESTIONS IN THE REVIEW\n"
                        f"{other_rqs_str}\n\n"
                    )
                    sys_prompt: str = Prompts().gen_meta_insights()

                    # Empty dict for fallback
                    fall_back = {
                        "paper_id": "",
                        "insight": []
                    }   
                    # call LLM
                    response_dict = utils.call_chat_completion(ai_model = self.ai_model,
                                                        llm_client = self.llm_client,
                                                        sys_prompt = sys_prompt,
                                                        user_prompt = user_prompt,
                                                        return_json = True, 
                                                        fall_back=fall_back)
                
                    response_dict["paper_id"] = paper_id
                    response_dict["question_id"] = rq_id
                    # Ensure insight key exists
                    if "insight" not in response_dict:
                        response_dict["insight"] = []
                    # Ensure insight is a list
                    if isinstance(response_dict["insight"], list):
                        pass
                    elif isinstance(response_dict["insight"], str):
                        response_dict["insight"] = [response_dict["insight"]]
                    else:
                        response_dict["insight"] = []
                    # Now append to the overall meta insights
                    meta_insights.append(response_dict)
                    with open(os.path.join(self.pickle_path, "meta_insights.pkl"), "wb") as f:
                        pickle.dump(meta_insights, f)
                
        
        # Convert to DataFrame
        meta_insights_df: pd.DataFrame = pd.DataFrame(meta_insights)
        
        # We want to eventually concat meta insights with insights, so we get all the columns neccesary to make meta insights compatible with insights
        # Make a temp copy of state.insights to drop unneccesary columns and then to merge with meta insights
        # Make copy
        temp_insights = deepcopy(self.state.insights)
        
        # Drop columns that will duplicate or are unneccesary
        cols_to_drop = [col for col in ["chunk_id", "insight"] if col in temp_insights.columns]
        temp_insights = temp_insights.drop(columns=cols_to_drop)
        
        # Drop duplicates so we have one row per (paper_id, question_id)
        temp_insights = temp_insights.drop_duplicates()

        # Merge meta insights into state.insights so meta insights have all the same columns as insights
        meta_insights_df = meta_insights_df.merge(
            temp_insights, how="left", on=["paper_id", "question_id"])

        # Prepare for exploding insights into separate rows
        meta_insights_df["insight"] = meta_insights_df["insight"].apply(self.ensure_list)
        # Explode meta insights so each insight is its own row
        meta_insights_df = meta_insights_df.explode("insight")
        # Create chunk_id column to identify meta insights
        meta_insights_df["chunk_id"] = [f"meta_insight_{pid}" for pid in meta_insights_df["paper_id"]]

        # Concat new meta insights
        self.state.insights = pd.concat(
            [self.state.insights, meta_insights_df], 
            ignore_index=True
        )
        
        # Add insight_id as i need this for joining in subsequent steps
        self.state.insights["insight_id"] = self.state.insights.index.astype(str)

        # Ensure chunk_id is string type - neccesary as earlier chunk ids were integers, now they have "meta insight_{paper_id}" strings too
        self.state.insights["chunk_id"] = self.state.insights["chunk_id"].astype(str)

        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "10_insights"))

        return meta_insights_df

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
    
    def recover_meta_insights_generation(self):
        print("Opening pickle file to recover meta insights generation...")
        with open(os.path.join(self.pickle_path, self.meta_insights_pickle_file), "rb") as f:
            recover_meta_insights = pickle.load(f)
        
        start = len(recover_meta_insights)
        print(f"Resuming meta insights generation from paper {start}...")
        self.get_meta_insights()
    
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
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                "download_status", "messy_question_id", "messy_paper_id",
                "ingestion_status", "chunk_id", "insight"
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
    def _strip_author_parenthetical(row):
        first_author = re.escape(row["paper_author"][0])
        # Remove (FirstAuthor ... ) greedily
        pattern = r"\(" + first_author + r".*?\)"
        # Always a list with one string
        if len(row["insight"]) > 0:
            insight_string = row["insight"][0]
        else: 
            return("")
        cleaned = re.sub(pattern, "", insight_string).strip()

        return cleaned

    def _gen_valid_embeddings_df(self):
        """
        Get the DataFrame of insights that are non-empty after stripping author parentheticals.
        Updates the self.state.insights with a new column 'no_author_insight_string'.
        Returns: pd.DataFrame
        """
        # Apply row-wise to your DataFrame
        self.state.insights["no_author_insight_string"] = self.state.insights.apply(self._strip_author_parenthetical, axis=1)

        out = self.state.insights[
            self.state.insights["no_author_insight_string"] != ""
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
                self.load_embeddings()
            elif recover == 'n':
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

    def load_embeddings(self):
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
            return(pd.NA)
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
            self.state.save(os.path.join(STATE_SAVE_LOCATION, "11_clusters"))
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
            self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "11_clusters"))
            return self.state.insights
        
            

class Summarize:
    def __init__(self,
                 state: Any,
                 llm_client: Any,
                 ai_model: str,
                 paper_output_length: int,  # Approximate total paper length in words
                 summary_save_location: str = None, 
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
        self.summary_save_location: str = summary_save_location or os.path.join(config.SUMMARY_SAVE_LOCATION, "parquet")
        self.state_save_location: str = state_save_location
        
        if not os.path.exists(insight_embedding_path):
            raise FileNotFoundError(f"Insight embeddings pickle not found at {insight_embedding_path}. Please run clustering first or amend the path to where you pickled your insight embeddings.")
        else:
            with open(insight_embedding_path, "rb") as f:
                self.insight_embeddings_array: np.ndarray = pickle.load(f)

        if not os.path.exists(self.summary_save_location):
            os.makedirs(self.summary_save_location, exist_ok=True)

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
    
    
    def summarize(self) -> Summaries:
        """
        Generate summaries for all clusters across all research questions.

        Returns:
            Summaries object containing a DataFrame of cluster summaries.
        """
        
        if os.path.exists(os.path.join(os.path.join(self.summary_save_location, "summaries.parquet"))):
            recover = None
            while recover not in ['r', 'n']:
                recover = input("Summaries already exist on disk. Do you want to recover (r) or generate new ones (n)? (r/n): ").lower()
            if recover == 'r':
                self.summaries = pd.read_parquet(os.path.join(self.summary_save_location, "summaries.parquet"))
                return self.summaries
            else:
                print("Re-running cleaning of summaries...")

        # We are going to send the insights to the LLM in the order of the shortest path, so that the most similar clusters are summarized close together
        # This will add coherence to the final paper when the summaries are stitched together
        # It will also aid in the applicaion of the sliding window for summary clean up
        shortest_paths = self._estimate_shortest_path()
        
        # Add calculated lengths to insights
        self.state.insights = self._calculate_summary_length()
        
        raw_summaries_list: List[str] = []

        total_clusters = len(self.state.insights.groupby("question_id")["cluster"].nunique(dropna=False))
        count = 1

        # Loop over unique research questions
        for rq_id in self.state.insights["question_id"].unique():
            rq_df: pd.DataFrame = self.state.insights[self.state.insights["question_id"] == rq_id].copy()
            rq_text: str = rq_df["question_text"].iloc[0]

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

                # Build user prompt for LLM
                user_prompt: str = (
                    f"Research question id: {rq_id}\n"
                    f"Research question text: {rq_text}\n"
                    "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
                    f"{'\n'.join(raw_summaries_list) if raw_summaries_list else ''}\n"
                    f"Cluster: {cluster}\n"
                    "INSIGHTS:\n" +
                    "\n".join(insights)
                )

                # Build system prompt from predefined method
                sys_prompt: str = Prompts().summarize(summary_length=length_str)

                messages: List[dict] = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Call LLM
                response: Any = self.llm_client.chat.completions.create(
                    model=self.ai_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )

                # Store raw JSON string from LLM
                raw_summaries_list.append(response.choices[0].message.content)

        # Parse JSON responses safely
        clean_summaries_list: List[dict] = []
        for idx, summary in enumerate(raw_summaries_list):
            try:
                clean_summaries_list.append(json.loads(summary))
            except json.JSONDecodeError:
                print(f"JSON decode failed for summary at index: {idx}")

        # Convert to DataFrame
        summaries_df: pd.DataFrame = pd.DataFrame(clean_summaries_list)


        print(
            f"Summaries saved here: {self.summary_save_location}\n"
            "Returned object is a Summaries instance. Access via `variable.summaries`.\n"
            f"Or load later with: Summaries.from_parquet('{self.summary_save_location}')"
        )

        self.summaries = summaries_df

        # Save summaries as this is LLM output we may want to reuse later - save as parquet
        os.makedirs(self.summary_save_location, exist_ok=True)
        self.summaries.to_parquet(os.path.join(self.summary_save_location, "summaries.parquet"))

        return summaries_df
    
    def identify_themes(self, save_file_name="summary_themes.parquet") -> pd.DataFrame:
        if not hasattr(self, "summaries"):
            raise ValueError("No summaries found. Please run summarize() first.")

        save_dir = self.summary_save_location
        save_path = os.path.join(save_dir, save_file_name)

        if os.path.exists(save_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input("Themed summaries already exist on disk. Recover (r) or generate new (n)? ").lower()
            if recover == 'r':
                self.summary_themes = pd.read_parquet(save_path)  # fix
                return self.summary_themes
            else:
                print("Re-running theming of summaries...")

        out_pdfs = []

        for question_id, rq_df in self.summaries.groupby("question_id", sort=False):
            question_text = rq_df["question_text"].iloc[0]
            summary_text = "\n\n".join(rq_df["summary"].tolist())

            user_prompt = (
                f"Research question id: {question_id}\n"
                f"Research question text: {question_text}\n"
                "SUMMARY TEXT:\n"
                f"{summary_text}\n"
            )
            sys_prompt = Prompts().llm_theme_id()

            fall_back = {"question_id": question_id, "themes": [], "other_bucket_rules": ""}

            resp = utils.call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                return_json=True,
                fall_back=fall_back,
            )

            # enforce schema even if empty
            themes = resp.get("themes") or []
            themes_df = pd.DataFrame(themes, columns=["id", "label", "criteria"])

            other_bucket_rules = (resp.get("other_bucket_rules") or "").strip()
            other_df = pd.DataFrame(
                [{"id": "other", "label": "Other", "criteria": other_bucket_rules}],
                columns=["id", "label", "criteria"],
            )

            out_row = pd.concat([themes_df, other_df], ignore_index=True)
            out_row["question_id"] = question_id
            out_row["question_text"] = question_text
            out_pdfs.append(out_row)

        output = pd.concat(out_pdfs, ignore_index=True, sort=False)

        self.summary_themes = output
        os.makedirs(save_dir, exist_ok=True)
        self.summary_themes.to_parquet(save_path)

        return self.summary_themes

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
        