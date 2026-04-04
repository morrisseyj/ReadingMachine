"""
Core pipeline components for ReadingMachine.

This module implements the primary analytical workflow used by
ReadingMachine to transform a corpus of documents into a structured
thematic synthesis. Each class corresponds to a conceptual stage in the
reading pipeline and operates by transforming the persistent pipeline
state objects (`CorpusState` and `SummaryState`).

The module is organized around four major processing stages:

    Ingestor
    Insights
    Clustering
    Summarize


Pipeline Overview
-----------------

The ReadingMachine pipeline converts a corpus of natural-language
documents into a structured thematic synthesis through the following
sequence of transformations:

    documents
    → chunks
    → insights
    → embeddings
    → clusters
    → cluster summaries
    → theme schemas
    → insight–theme mappings
    → populated themes
    → orphan integration
    → redundancy reduction


Classes
-------

Ingestor
    Reads source documents (PDF or HTML) and converts them into the
    structured corpus representation used by the pipeline. This stage
    populates the `CorpusState.full_text` table and ensures document
    metadata is correctly associated with paper identifiers.

Insights
    Extracts atomic insights from the corpus using a two-pass reading
    process:

        • chunk-level insight extraction
        • whole-document (meta) insight extraction

    Insights represent the smallest analytical unit of the pipeline and
    serve as the basis for all downstream organization and synthesis.

Clustering
    Generates vector embeddings for insights and groups them into
    provisional clusters using dimensionality reduction and density-
    based clustering.

    Clusters are used only as an organizational scaffold that helps
    structure the first pass of theme generation. They are not treated
    as analytical conclusions and do not determine the final thematic
    structure.

Summarize
    Performs the thematic synthesis process. This stage transforms
    clustered insights into narrative thematic summaries through a
    controlled multi-step workflow:

        1. Cluster summarization
        2. Theme schema generation
        3. Insight-to-theme mapping
        4. Theme population
        5. Orphan detection and reintegration
        6. Redundancy reduction

    These steps may be iterated to refine the thematic structure before
    producing the final synthesis output.


State Architecture
------------------

All classes in this module operate on two persistent state objects
defined in `readingmachine.state`:

CorpusState
    Stores the structured representation of the corpus including
    documents, chunks, and extracted insights.

SummaryState
    Records the evolving artifacts of the synthesis stage such as
    cluster summaries, theme schemas, mappings, and theme summaries.

This separation preserves traceability between the original text and
the final synthesis.


Design Principles
-----------------

The pipeline is designed around several methodological principles:

Atomic insight extraction
    Insights represent the smallest analytical units in the corpus.

Traceability
    Every synthesized claim remains linked to its source document and
    text segment.

Iterative synthesis
    Theme structures are refined through repeated schema generation and
    orphan reintegration.

Resumable computation
    Long-running LLM operations persist intermediate results so that
    interrupted processes can safely resume.

Separation of reading and synthesis
    Corpus processing (`Ingestor`, `Insights`, `Clustering`) is kept
    distinct from interpretive synthesis (`Summarize`).


Usage
-----

The pipeline is typically executed as a sequence of class
instantiations that progressively transform the pipeline state:

    corpus = CorpusState.load(...)

    ingestor = Ingestor(...)
    ingestor.ingest_papers()

    insights = Insights(...)
    insights.get_chunk_insights()
    insights.get_meta_insights()

    clustering = Clustering(...)
    clustering.embed_insights()
    clustering.reduce_dimensions()
    clustering.generate_clusters(...)

    summarizer = Summarize(...)
    summarizer.summarize_clusters()
    summarizer.gen_theme_schema()
    summarizer.map_insights_to_themes()
    summarizer.populate_themes()
    summarizer.address_orphans()
    summarizer.address_redundancy()

Each stage updates the pipeline state, allowing intermediate artifacts
to be inspected or reused across runs.
"""
# import custom libraries

from . import config, utils
from .state import CorpusState, SummaryState
from .prompts import Prompts

# import standard libraries
from typing import List, Any, Optional
import pandas as pd
import numpy as np
from copy import deepcopy
import os
from pathlib import Path
from collections import defaultdict
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
import math


class Ingestor:
    """
    Document ingestion, metadata extraction, and deduplication stage of the ReadingMachine pipeline.

    The `Ingestor` class reads source documents from a filesystem directory
    and converts them into the structured corpus representation used by the
    rest of the pipeline. This stage transforms raw files into entries in the
    `CorpusState.full_text` table and ensures that each document is linked to
    the correct research question and paper identifier.

    Supported file formats
    ----------------------
    - PDF
    - HTML

    HTML files are cleaned using structural parsing and then optionally
    processed by an LLM to extract the main textual content of the page.

    After ingestion, the class performs **metadata extraction** using an LLM.
    This step populates the following fields for each paper:

    - paper_title
    - paper_author
    - paper_date

    These values are extracted from the beginning of the document text and
    validated for type consistency before being written back into
    `corpus_state.insights`.

    Deduplication
    -------------
    Following metadata extraction, the ingestion stage performs a **global
    deduplication pass** over the corpus.

    Deduplication occurs in two steps:

        1. Exact duplicate removal
           Records with identical author–title–year combinations are removed.

        2. Fuzzy duplicate detection (human-in-the-loop)
           Records with highly similar metadata are exported for manual review.
           The user removes duplicates and confirms a final deduplicated set.

    After user confirmation, the corpus is filtered so that only unique
    `paper_id` values are retained across:

        - `corpus_state.full_text`
        - `corpus_state.insights`

    Importantly, this process does **not modify the user's filesystem**.
    Duplicate files may remain on disk, but they are excluded from all
    downstream processing.

    Pipeline role
    -------------
    The ingestion stage produces three key artifacts:

        corpus_state.full_text
        corpus_state.insights (metadata-complete and deduplicated)
        corpus_state.questions (canonicalized)

    These artifacts are later used by downstream stages such as:

    - chunking
    - insight extraction
    - clustering
    - thematic synthesis

    Deduplication ensures that:

    - duplicate documents are not reprocessed
    - token usage and runtime are minimized
    - insights are not artificially duplicated during synthesis

    Design principles
    -----------------
    The ingestion logic is intentionally strict about identifier consistency.

    Document filenames are expected to correspond to the `paper_id` values
    present in the insights table. Duplicate filenames or mismatches between
    the corpus metadata and filesystem contents will trigger warnings or
    user confirmation prompts.

    Deduplication is performed **after metadata extraction** because metadata
    provides the canonical basis for identifying duplicate documents.

    Attributes
    ----------
    corpus_state : CorpusState
        Working corpus state containing literature metadata and insight
        records. This object is mutated during ingestion.

    file_path : str
        Directory containing the source documents to ingest.

    llm_client : Any
        LLM client used for HTML parsing and metadata extraction.

    ai_model : str
        Model name used when making LLM calls.

    ingestion_errors : List[str]
        List of files that failed ingestion due to parsing errors.

    pickle_path : str
        Directory used to persist intermediate metadata extraction results
        for resume support.

    fuzzy_check_path : str
        Directory used to store fuzzy duplicate review files for
        manual deduplication.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_model: str,
        corpus_state: Optional[CorpusState] = None,
        questions: Optional[List[str]] = None,
        papers: Optional[pd.DataFrame] = None,
        file_path: str = os.path.join(os.getcwd(), config.CORPUS_LOCATION),
        pickle_path: str = config.PICKLE_SAVE_LOCATION, # For storing the pickles of LLM metadata retreival for resume
        fuzzy_check_path: str = config.FUZZY_CHECK_PATH
    ) -> None:
        """Initialize Ingestor and validate corpus_state/papers format."""

        self.RUN = "ingest"
        self.fuzzy_check_path = fuzzy_check_path
    
        self.corpus_state = deepcopy(
            utils.validate_format(
                corpus_state=corpus_state,
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

        self.corpus_state.enforce_canonical_question_text()
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
        Discover ingestible documents in the configured directory.

        This method recursively searches the ingestion directory for files
        with supported extensions (`.pdf` or `.html`). It also checks for
        duplicate filenames across subdirectories.

        Because `paper_id` values are derived from filenames, duplicate
        filenames would produce conflicting identifiers during ingestion.
        When duplicates are detected, the method raises an error and reports
        the full paths of the conflicting files so the user can resolve them.

        Returns
        -------
        List[str]
            Absolute paths to all ingestible documents discovered in the
            directory tree.

        Raises
        ------
        ValueError
            If duplicate filenames are detected in the ingestion directory.
        
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
        """
        Extract text from a PDF document.

        Each page of the document is extracted separately using PyMuPDF.
        The resulting list preserves page boundaries, which can be useful
        for downstream debugging or metadata extraction.

        Parameters
        ----------
        path : str
            Absolute path to the PDF file.

        Returns
        -------
        List[str]
            List of page-level text strings.
        """
        with pymupdf.open(path) as doc:
            return [doc[i].get_text() for i in range(doc.page_count)]

    @staticmethod
    def _html_cleaner(html_content: str) -> str:
        """
       Remove structural noise from raw HTML.

        The function removes common layout elements such as navigation,
        scripts, and style tags before extracting visible text from the
        `<body>` element.

        Parameters
        ----------
        html_content : str
            Raw HTML content.

        Returns
        -------
        str
            Cleaned plain-text representation of the document.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)
        return ""

    @staticmethod
    def _html_chunker(clean_html: str, token_limit: int = 16000) -> List[str]:
        """
        Split cleaned HTML text into segments suitable for LLM processing.

        Large HTML documents may exceed the token limits of the model used
        for HTML parsing. This function divides the cleaned text into chunks
        that can be safely processed by the LLM.

        Parameters
        ----------
        clean_html : str
            Plain-text HTML content.

        token_limit : int
            Maximum character length per chunk.

        Returns
        -------
        List[str]
            List of text segments to be processed by the LLM.
        """
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
        """
        Use an LLM to extract meaningful content from HTML segments.

        Some HTML pages contain large amounts of structural noise that cannot
        be reliably removed with rule-based parsing alone. This method sends
        HTML segments to the LLM along with a prompt instructing the model to
        extract the main textual content.

        Parameters
        ----------
        html_list : List[str]
            List of HTML text segments to process.

        prompt : str
            System prompt used to guide the extraction.

        Returns
        -------
        List[str]
            Cleaned textual segments returned by the model.
        """
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
        """
        Dispatch ingestion based on file type.

        This method determines the file format and routes the document to
        the appropriate ingestion routine.

        Supported formats:

        - PDF → parsed with PyMuPDF
        - HTML → cleaned, chunked, and optionally processed by the LLM

        Parameters
        ----------
        file_full_path : str
            Absolute path to the document.

        Returns
        -------
        List[str]
            List of text segments representing the document contents.
        """
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
        Ingest all documents in the configured directory.

        This method performs the primary ingestion workflow:

        1. Discover files in the ingestion directory
        2. Parse each document into text
        3. Track ingestion success or failure
        4. Match ingested documents with expected `paper_id` values
        5. Construct the `full_text` corpus table
        6. Clean the insights table of unmatched or unused records

        The resulting full text representation is stored in
        `corpus_state.full_text`.

        Returns
        -------
        pd.DataFrame
            Updated insights table with only successfully ingested papers.

        Notes
        -----
        The method performs several integrity checks:

        - documents without matching IDs are flagged
        - expected documents without files are reported
        - ingestion failures are tracked in `.ingestion_errors`

        User confirmation prompts are used when mismatches occur to
        prevent accidental loss of records.
        
        """
        
        working_insights = (
            self.corpus_state.insights.copy()
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
                print(f"Error ingesting file {file}: {e}")
                list_of_papers_by_page.append([str(e)])
                self.ingestion_errors.append(file)
                ingestion_status.append(0)

        # Confirm ingestion errors
        if self.ingestion_errors:
            abort_failed_ingest = None
            while abort_failed_ingest not in ["y", "n"]:
                abort_failed_ingest = input(
                    "Ingestion errors occurred. Examine .ingestion_errors and corpus_state.full_text.\n"
                    "Hit 'c' to confirm having read this message:\n\n\n"
                ).lower()
            
            if abort_failed_ingest == "y":
                print("Aborting ingestion. Please review the ingestion errors (returned below and accessible via .ingestion_errors) and the corpus_state.full_text object to see which papers were not ingested successfully.\n\n\n")
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

            unmatched_files = failed_id_matches[
                failed_id_matches["ingestion_ids_matched"] == "import file does not match possible id"
            ]

            missing_files = failed_id_matches[
                failed_id_matches["ingestion_ids_matched"] == "existing id had no corresponding file"
            ]


            abort_failed_match = None
            while abort_failed_match not in ['y', 'n']:
                abort_failed_match = input(
                "\n\nWarning: File / metadata ID mismatches detected.\n\n"
                f"Files ingested that did not match an existing paper_id: {len(unmatched_files)}\n"
                f"Insights rows with no corresponding file: {len(missing_files)}\n\n"
                "If you want files to link to specific questions, place them inside folders.\n"
                "If you invoked the getlit.py module in this library this warning IS likely relevant to you.\n" \
                "If you are reading your own corpus, you can likely ignore this message.\n\n"
                "If you ignore this warning any paper ids that did not have a matching file will be deleted from corpus_state.insights. You can look these up later by exploring the corpus_state.insights object created earlier in the pipeline.\n\n"    
                "Do you wish to abort ingestion to review the failed id matches? (y/n):\n"
            ).lower()
            
            if abort_failed_match == 'y':
                print("Aborting ingestion. Please review the failed id matches (returned below and accessible via .failed_id_matches).\n\n")
                return(failed_id_matches)
            else:
                pass #continue with ingestion despite failed id matches


        # Identify all undownloaded files for the user to see what they are not getting. 
        # Note not tracking these is a design. This package manages reading. It has a module that helps with identifying papers to read, but it is the users responsibility to get the papers. 
        # So we record this error here but it is not persisted to corpus_state
        self.dropped_papers = working_insights[working_insights["ingestion_status"] == 0]

        # Get the all the ingested papers
        working_insights = working_insights[working_insights["ingestion_status"] == 1] # Filter to just the papers that were ingested successfully, as these are the ones we have insights for and can track through the pipeline.
        # Populate the full text corpus_state object
        full_text = (
            working_insights[["paper_id", "pages"]]
            .assign(full_text=lambda x: [" ".join(pages) for pages in x["pages"]])
            .drop(columns=["pages"]) # Drop pages as they take up memory and are not needed, we have the full text now
        )
        # Set the full text as a corpus_state attribute
        self.corpus_state.full_text = full_text

        # Drop pages from insights as well as other fields from the lit retrieve module that are no longer needed:
        working_insights.drop(columns=["pages", "download_status", "messy_question_id", "messy_paper_id"], inplace=True)

        # Set as corpus_state attributes
        self.corpus_state.insights = working_insights

        if self.dropped_papers.shape[0] > 0: # Set as none on init, gets created if there are dropped papers
                print(
                    f"Warning: {self.dropped_papers.shape[0]} paper(s) were not downloaded from your original list. These papers are listed in the .dropped_papers attribute.\n"
                    "You can review these papers to see what was not ingested successfully, update and potentially re-ingest them, but they will not be included in the rest of the pipeline as we have no text to work with for these papers.\n\n"
                    )
                
        print("\nPaper ingestion complete")

    def _get_metadata_from_llm(self, paper_id: str, text: str) -> dict[str, Any]:
        """
        Extract publication metadata from a document using an LLM.

        The model is provided with the beginning of the document text
        (typically the first few pages) and asked to identify:

        - paper_title
        - paper_author
        - paper_date

        Parameters
        ----------
        paper_id : str
            Identifier of the paper being processed.

        text : str
            Portion of the document text used for metadata extraction.

        Returns
        -------
        dict[str, Any]
            Dictionary containing extracted metadata fields.

        Raises
        ------
        KeyError
            If the LLM response is missing required metadata keys.
        """
        
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
            """
            Normalize and validate metadata values returned by the LLM.

            LLM responses can contain inconsistent or unexpected data types
            (e.g., strings, lists, dictionaries, or malformed values). This
            helper function coerces metadata values into the expected type
            while safely handling missing or invalid entries.

            The function applies the following rules:

            - Explicit missing values (`NA`, empty strings, or pandas NA)
            are converted to `pd.NA`.
            - If a string is provided and the desired type is `int`,
            the function attempts to parse the string as an integer
            (used primarily for publication year).
            - If the value already matches the desired type, it is
            returned unchanged.
            - Unexpected structures (lists, dictionaries, etc.) are
            treated as missing values and converted to `pd.NA`.

            Parameters
            ----------
            x : Any
                Metadata value returned by the LLM.

            desired_type : type
                Expected Python type for the metadata field. Currently
                used with `str` (title, author) and `int` (publication year).

            Returns
            -------
            Any
                A cleaned value matching the desired type, or `pd.NA`
                if the input cannot be safely converted.

            Notes
            -----
            This function exists to protect the integrity of the corpus
            metadata tables. LLM outputs are inherently probabilistic and
            may occasionally return unexpected formats; treating such values
            as missing prevents downstream type errors and preserves the
            stability of joins and aggregations later in the pipeline.
            """
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
        
        """
        Populate document metadata using LLM extraction.

        This method iterates through papers in the corpus and extracts
        metadata fields using `_get_metadata_from_llm`.

        Intermediate results are written to a pickle file so the process
        can be resumed if interrupted.

        Parameters
        ----------
        metadata_check_df : pd.DataFrame
            DataFrame containing paper identifiers and text used for
            metadata extraction.

        recovered_metadata_check : List[pd.DataFrame], optional
            Previously completed metadata extraction results used when
            resuming a partially completed run.

        Returns
        -------
        List[pd.DataFrame]
            List of metadata records, one per processed paper.
        """
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
            
            # Save to pickle to allow for resume, make sure path exists - use safe pickle to ensure atomic save
            os.makedirs(self.pickle_path, exist_ok=True)
            utils.safe_pickle(output, os.path.join(self.pickle_path, "metadata_check.pkl"))

        return(output)

    def update_metadata(self) -> pd.DataFrame:
        """
        Update paper metadata using LLM extraction.

        This method identifies papers with missing or incomplete metadata
        and attempts to populate the following fields:

        - paper_title
        - paper_author
        - paper_date

        The metadata is extracted from the beginning of each document's
        full text and then merged back into `corpus_state.insights`.

        Resume support
        --------------
        If a previous metadata extraction run was interrupted, the method
        can resume from a stored pickle file containing partial results.

        Returns
        -------
        pd.DataFrame
            Updated insights DataFrame containing the completed metadata.
        """
        #Create the metadata check which is the dataframe containing the columns i need for the check
        metadata_check_df = (
            self.corpus_state.insights.copy()[["paper_id", "paper_title", "paper_author", "paper_date"]]
            .merge(self.corpus_state.full_text[["paper_id", "full_text"]],
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

        # Merge back to corpus_state.insights. Drop the old metadata columns and replace with the new metadata from the llm. This ensures that any metadata that was missing or incorrect is updated with the llm response, while any existing correct metadata is retained. We merge on paper_id to ensure we are updating the correct records, and we validate one_to_one to ensure there are no duplicate paper_ids which would indicate an issue with the data integrity.
        updated_insights = (
            self.corpus_state.insights
            .drop(columns=["paper_title", "paper_author", "paper_date"])
            .merge(full_meta_data_df, how="left", on="paper_id", validate="one_to_one")
        )

        # Update the corpus_state.insights to now have the correct metadata. 
        # Note we don't save here as its not the end of the object and we have the pickle to handle recovery if we need to re run.
        self.corpus_state.insights = updated_insights

        return self.corpus_state.insights
    
    
    def drop_duplicates(self, threshold = 0.9) -> pd.DataFrame:
        """
         Perform the duplicate detection stage of the pipeline and generate a review file.

        This method identifies potential duplicate documents using full-text similarity
        (shingles + Jaccard) and prepares a CSV file for manual review. It represents
        the "deduplication step" in the pipeline, but does NOT modify the corpus state.

        The user is expected to:
        1. Open the generated CSV file
        2. Review grouped duplicates (`sim_group`)
        3. Remove duplicate rows (keeping one per document)
        4. Run `update_state()` to apply the changes

        Parameters
        ----------
        threshold : float, default=0.9
            Similarity threshold for grouping documents using Jaccard similarity.
            Must be between 0 and 1. Higher values result in stricter duplicate detection.

        Returns
        -------
        None
            This function does not return a DataFrame. It writes a CSV file for manual review.

        Side Effects
        ------------
        - Writes a CSV file to:
        `self.fuzzy_check_path / self.RUN / duplicate_check.csv`
        - Prints instructions for completing the deduplication workflow

        Notes
        -----
        - This function is part of a prescriptive CLI pipeline:
        detection → manual review → update_state
        - The corpus state remains unchanged until `update_state()` is called
        - Uses full-text similarity for high-precision duplicate detection
        """
        # Set the save path for use to insepct
        save_dir = os.path.join(self.fuzzy_check_path, self.RUN)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "duplicate_check.csv")

        df_for_review = utils.prepare_dedup_review(state = self.corpus_state, threshold=threshold, engine="shingles")
        df_for_review.to_csv(output_path, index=False)

        print(
            f"Potential duplicate review file saved to {output_path}.\n"
            "Delete duplicate rows and keep one per paper.\n"
            "Then run update_state()."
        )

        return None

    
    # def drop_duplicates(self) -> pd.DataFrame:
    #     """
    #     Remove exact duplicates from insights using metadata.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Deduplicated DataFrame.
    #     """

    #     dropped_exact = utils.drop_exact_duplicates(self.corpus_state.insights)

    #     # Store for fuzzy pass
    #     self.dropped_exact_dupes = dropped_exact.copy()

    #     return self.dropped_exact_dupes

    # def drop_fuzzy_duplicates(self, similarity_threshold: int = 90) -> pd.DataFrame:
    #     """
    #     Generate fuzzy duplicate candidates for manual review.

    #     Returns
    #     -------
    #     None
    #     """

    #     if not hasattr(self, "dropped_exact_dupes"):
    #         raise ValueError("Run drop_duplicates() before drop_fuzzy_duplicates().")

    #     review_df = utils.prepare_fuzzy_review_df(
    #         self.dropped_exact_dupes,
    #         similarity_threshold=similarity_threshold
    #     )

    #     # Create run-specific folder
    #     save_dir = os.path.join(self.fuzzy_check_path, self.RUN)
    #     os.makedirs(save_dir, exist_ok=True)

    #     output_path = os.path.join(save_dir, "fuzzy_matches.csv")
    #     review_df.to_csv(output_path, index=False)

    #     print(
    #         f"Fuzzy duplicate review file saved to {output_path}.\n"
    #         "Delete duplicate rows and keep one per paper.\n"
    #         "Then run update_state()."
    #     )

    #     return None

    def update_state(self, 
                     filename: str, 
                     encoding: str = "utf-8",
                     output_cols = [
                        "question_id",
                        "question_text",
                        "search_string_id",
                        "search_string",
                        "paper_id",
                        "paper_title",
                        "paper_author",
                        "paper_date",
                        "doi"
                        ]
                    ) -> pd.DataFrame:
        """
        Updates the insights df after the user has manually removed fuzzy duplicate papers from the cvs/xlsx file generated in the drop_fuzzy_duplicates function.

        Becuase this is done after ingestion (because complete metadata is used for duplicate checking and that is done by LLM on the full_text) 
        the insights and full_text elements of the state stand to be out of sync. This function realigns them and updates the state

        Parameters
        ----------
        filename : str
            Name of the manually reviewed file.

        encoding : str
            File encoding for reading the reviewed file.

        output_cols : list, optional
            List of columns to include in the output DataFrame. If empty, all columns are included.

        Returns
        -------
        pd.DataFrame
            Cleaned insights DataFrame.
        """
        # Set the filepath to the reviewed file based on the provided filename and the expected location of the fuzzy check outputs. 
        filepath = os.path.join(self.fuzzy_check_path, self.RUN, filename)

        # Check file exists
        if not os.path.isfile(filepath):
            raise ValueError(f"Reviewed file not found at expected location: {filepath}. Please ensure the file is in the correct directory and the filename is correct.")
        
        #Get the updated insights df with the correct schema
        insights_df = CorpusState.load_insights_from_csv_xslx(filepath=filepath, encoding=encoding, output_cols=output_cols)

        # Clean the author field:
        if "paper_author" in insights_df.columns:
            insights_df["paper_author"] = insights_df["paper_author"].replace(
                ["", "No author found", "NA", "null", pd.NA, np.nan],
                None
            )

        # Align with full text:
        # Get unique valid paper ids from the deduped insights
        valid_ids = set(insights_df["paper_id"])
        # Sanity check to make sure some papers came in
        if len(valid_ids) == 0:
            raise ValueError(f"No records found in {filepath}. Please ensure you have kept at least one record per paper and that the paper_id column is intact.")
        
        # Make temp copy of full text to operate over
        temp_full_text = self.corpus_state.full_text.copy()

        # Filter full text
        temp_full_text = (
           temp_full_text[
                temp_full_text["paper_id"].isin(valid_ids)
            ].copy()
        )

        # Filter insights - to make sure only unique ids
        insights_df = (
            insights_df[
                insights_df["paper_id"].isin(valid_ids)
            ].copy()
        )

        # reassign to corpus state
        self.corpus_state.full_text = temp_full_text
        self.corpus_state.insights = insights_df

        # Note we don't save as we are now going to chunk this in the same class
        # Return for inspection
        return self.corpus_state.insights

    @staticmethod
    def _drop_duplicate_chunks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate text chunks within each paper.

        This function deduplicates chunks at the (paper_id, chunk_text) level,
        ensuring that identical text segments extracted multiple times (e.g.,
        due to PDF parsing artifacts or layout duplication) are only processed once.

        This is a low-risk operation: only exact matches are removed, so no
        meaningful narrative content is lost.

        Parameters
        ----------
        df : pd.DataFrame
            Chunk-level DataFrame. Must contain:
            - 'paper_id' : identifier for each document
            - 'chunk_text' : extracted text content for each chunk

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicate (paper_id, chunk_text) rows removed.
            Index is reset.

        Raises
        ------
        ValueError
            If required columns are missing.

        Notes
        -----
        - Deduplication is performed *within* papers, not across papers.
        - This step reduces redundant LLM calls and improves efficiency.
        """
        if not set(["paper_id", "chunk_text"]).issubset(df.columns):
            raise ValueError("Input DataFrame must contain 'paper_id' and 'chunk_text' columns.")
        
        df = df.drop_duplicates(subset=["paper_id", "chunk_text"])
        return df.reset_index(drop=True)
    
    @staticmethod
    def _drop_boilerplate(df) -> pd.DataFrame:
        """
        Remove high-frequency repeated chunks within each paper.

        This function identifies chunks that appear repeatedly within the same
        document (e.g., headers, footers, page titles) and removes them. These
        elements are typically structural boilerplate and do not contribute
        meaningful semantic content.

        A chunk is considered boilerplate if it appears more than 10 times
        within a single paper.

        Parameters
        ----------
        df : pd.DataFrame
            Chunk-level DataFrame. Must contain:
            - 'paper_id'
            - 'chunk_text'

        Returns
        -------
        pd.DataFrame
            DataFrame with high-frequency repeated chunks removed.
            Index is reset.

        Notes
        -----
        - Filtering is applied at the (paper_id, chunk_text) level.
        - This does NOT remove entire papers, only repeated text segments.
        - The threshold (10) is intentionally conservative to avoid removing
        legitimate repeated content.
        """
        counts = df.groupby(["paper_id", "chunk_text"])["chunk_text"].transform("size")
        boilerplate = df[counts <= 10].reset_index(drop=True)
        return boilerplate

    @staticmethod
    def _drop_extreme_table_chunks(df):
        """
        Remove chunks that are highly likely to represent tabular or non-narrative content.

        This function filters out chunks dominated by numeric or structurally
        fragmented content (e.g., table rows, data grids) while preserving
        narrative text. The filtering logic is conservative and designed to
        minimize loss of meaningful prose.

        A chunk is removed if it:
        - Contains a high proportion of numeric tokens (digit_ratio > 0.5), OR
        - Is short and lacks lexical richness (few longer words), AND does not
        resemble a sentence.

        Chunks are retained if they exhibit sentence-like structure (e.g.,
        contain punctuation or sufficient lexical complexity), even if they
        partially resemble table content.

        Parameters
        ----------
        df : pd.DataFrame
            Chunk-level DataFrame. Must contain:
            - 'chunk_text'

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with extreme table-like chunks removed.

        Notes
        -----
        - This function prioritizes recall of narrative text over aggressive filtering.
        - Designed to handle structured PDF extractions (e.g., MuPDF output).
        - Filtering is heuristic-based and language-agnostic.
        - Applied after deduplication and boilerplate removal for efficiency.
        """

        def is_extreme_table(text):
            words = text.split()

            digit_ratio = sum(any(c.isdigit() for c in w) for w in words) / max(len(words), 1)

            # Strong signal: mostly numeric
            if digit_ratio > 0.5:
                return True
    
            # Weak signal: short + low lexical richness
            if len(words) < 15:
                long_words = sum(len(w) > 4 for w in words)
                if long_words < 3:
                    return True
            
            return False

        def has_sentence_structure(text):
            return (
                "." in text or
                len([w for w in text.split() if len(w) > 4]) > 5
                )

        working_df = df.copy()
        texts = working_df["chunk_text"]

        has_sentence = texts.apply(has_sentence_structure)
        is_table = texts.apply(is_extreme_table)

        working_df = working_df[has_sentence | ~is_table]

        return working_df

    def chunk_papers(
        self,
        chunk_size: int = 3500,
        chunk_overlap: int = 350,
        length_function=len,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False
    ) -> None:
        """
        Split full-text documents into overlapping chunks and apply cleaning steps.

        This method:
        1. Splits each document in `corpus_state.full_text` into smaller text chunks
        using a recursive character-based splitter.
        2. Flattens the resulting nested structure into a chunk-level DataFrame.
        3. Assigns unique chunk IDs.
        4. Applies a series of cleaning operations:
            - exact deduplication
            - boilerplate removal (high-frequency repeated chunks)
            - removal of extreme table-like content
        5. Updates `corpus_state.chunks` with the cleaned chunks.
        6. Rebuilds `corpus_state.insights` to align with the cleaned chunk set.
        7. Saves the updated corpus state.

        Parameters
        ----------
        chunk_size : int, default=3500
            Maximum size of each chunk (in characters or as defined by `length_function`).

        chunk_overlap : int, default=350
            Number of overlapping characters between consecutive chunks.

        length_function : callable, default=len
            Function used to measure chunk length.

        separators : list of str, optional
            Ordered list of separators used for recursive splitting.
            Defaults to ["\n\n", "\n", ". ", "! ", "? ", " ", ""].

        is_separator_regex : bool, default=False
            Whether separators should be treated as regex patterns.

        Returns
        -------
        None

        Side Effects
        ------------
        - Updates:
            - `self.corpus_state.chunks`
            - `self.corpus_state.insights`
        - Persists updated state to disk.

        Notes
        -----
        - `full_text` remains unchanged and acts as the source of truth.
        - Cleaning steps are intentionally conservative to preserve narrative content.
        - The pipeline is designed to reduce token usage and improve LLM efficiency
           without sacrificing semantic coverage.
        """

        def _normalize_text(text: str) -> str:

            if not isinstance(text, str):
                return text

            # remove soft hyphens
            text = text.replace('\xad', '')

            # fix broken words
            text = re.sub(r'(?<=\w)\n(?=\w)', '', text)

            # convert multiple newlines to paragraph markers
            text = re.sub(r'\n{2,}', '\n\n', text)

            # convert single newlines to space
            text = re.sub(r'\n', ' ', text)

            # normalize whitespace
            text = re.sub(r'\s+', ' ', text)

            return text.strip()

        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            is_separator_regex=is_separator_regex
        )

        full_text_list = (
            self.corpus_state.full_text["full_text"]
            .fillna("")
            .apply(_normalize_text)
            .to_list()
        )

        chunks_list: List[List[str]] = [text_splitter.split_text(text) for text in full_text_list]
        # Create the chunks corpus_state from the full_text corpus_state
        self.corpus_state.full_text["chunk_text"] = chunks_list
        self.corpus_state.chunks = self.corpus_state.full_text[["paper_id", "chunk_text"]].explode("chunk_text").reset_index(drop=True).copy()
        self.corpus_state.chunks["chunk_id"] = [f"chunk_{i+1}" for i in range(self.corpus_state.chunks.shape[0])]

        # Chunks from full_text as its now joined by paper and question id
        self.corpus_state.full_text.drop(columns=["chunk_text"], inplace=True)

        # Now clean up the chunks by dropping duplicates, boilerplayte and tables
        temp_chunks = self.corpus_state.chunks.copy()
        # Remove NA to avoid crashing on split in the functions
        temp_chunks = temp_chunks[temp_chunks["chunk_text"].notna()]
        # Now call the cleaning functions
        temp_chunks = self._drop_duplicate_chunks(temp_chunks)
        temp_chunks = self._drop_boilerplate(temp_chunks)
        temp_chunks = self._drop_extreme_table_chunks(temp_chunks)

        self.corpus_state.chunks = temp_chunks.reset_index(drop=True).copy()

        # Get chunk_id into insights
        temp_insights = self.corpus_state.insights.copy()
        self.corpus_state.insights = (
            self.corpus_state.chunks
            .drop(columns=["chunk_text"])
            .merge(temp_insights, how="left", on="paper_id")
        )

        # Save the updated corpus_state
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "07_full_text_and_chunks"))

    def chunk_sanity_check(self):
        """
        Perform a sanity check on the chunking process.
        
        Calculates:
            The number of chunks per paper
            The average word length of the chunks per paper(approximate)
            The min and max chunk length in words per paper
        
        Returns:
           pd.DataFrame:
                A DataFrame summarizing the chunking statistics for each paper
        
        Parameters:
            None

        """
        # Get the first few records of the chunks corpus_state to inspect

        temp_chunks = self.corpus_state.chunks.copy()

        temp_chunks["chunk_len"] = temp_chunks["chunk_text"].str.split().str.len()

        summary = (
            temp_chunks
            .groupby("paper_id")
            .agg(
                num_chunks=("chunk_id", "nunique"),
                avg_chunk_length=("chunk_len", "mean"),
                min_chunk_length=("chunk_len", "min"),
                max_chunk_length=("chunk_len", "max"),
                total_words=("chunk_len", "sum")
            )
            .reset_index()
        )

        summary.sort_values("num_chunks", ascending=False, inplace=True)

        return summary

class Insights:
    def __init__(
        self,
        corpus_state: "CorpusState",
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
            corpus_state (CorpusState): 
                Container for all relevant corpus_state data including chunks, 
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


        # Ensure corpus_state has all required columns before processing
        self.corpus_state = deepcopy(
            utils.validate_format(
            corpus_state=corpus_state,
            questions=None,
            injected_value=None,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi"
            ],
            injected_required_cols=None
            )
        )

        self.corpus_state.enforce_canonical_question_text()

    

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
                with existing corpus_state metadata and assigned back to
                `self.corpus_state.insights`.

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
        #    insights == []     → valid recovery corpus_state but zero chunks processed
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

        remaining_chunks = self.corpus_state.chunks[
            ~self.corpus_state.chunks["chunk_id"].isin(processed_chunks)
        ]

        # Generate a list of all the research questions with ids in the form <rq_id>: <rq_text> for the llm to consdier against each chunk
        rqs_ids = [f"{row['question_id']}: {row['question_text']}" for _, row in self.corpus_state.questions.iterrows()]
        rqs_ids_str = "\n".join(rqs_ids)

        # get unique paper metadata to append to chunks so that insights can be cited
        paper_metadata = (
            self.corpus_state.insights
            [["paper_id", "question_text", "paper_author", "paper_date"]]
            .sort_values("paper_author", na_position="last")
            .drop_duplicates("paper_id")
        )

        # Merge chunk text with metadata (author, date, etc.)
        temp_state_insights: pd.DataFrame = remaining_chunks.merge(
            paper_metadata[["paper_id", "question_text", "paper_author", "paper_date"]],
            how="left",
            on=["paper_id"],
            validate="many_to_one"
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
            
            # Turn the response into a df with columns question_id and insight, where each row is a different insight, and the insights are the insights for that question id that were extracted from the chunk. This will make it easier to merge into the corpus_state later.
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
            # Batch writes every 10 chunks
            if (idx + 1) % 10 == 0:
                utils.safe_pickle(insights, os.path.join(self.pickle_path, self.chunk_insights_pickle_file))

        # Convert insights list to DataFrame
        print("Converting insights to DataFrame and merging into corpus_state...")
        insights_complete_df: pd.DataFrame = pd.concat(insights).reset_index(drop=True) if insights else pd.DataFrame(columns=["question_id", "insight", "chunk_id", "paper_id"])
        
        # Merge into global insights table - first drop existing insight columns if present (these can be created by previous runs of recover_chunk_insights_generation)
        base_insights = (
            self.corpus_state.insights
            .drop(columns=["insight"], errors="ignore") # drop the metadata columns if they exist as we will merge them back in from the corpus_state later, but ignore if they don't exist as this function can be run multiple times and they will only be there after the first run
        )
        
        # Now we merge this chunk insights with all the insights data and metadata. 
        # Notably we drop question_id from the corpus_state.insights as previously if papers wwere not associated with a question when importing them, the question_id was NA
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

        # Assign to corpus_state
        self.corpus_state.insights = working_insights_df

        # Note i don't save here as i only save at the end of the class's operations. This is cleaner for the user. 
        # Also since i have the recover function in place, once the insights are generated for this class it should be quick to recreate to this point
        return self.corpus_state.insights
    
    
    def _recover_chunk_insights_generation(self):
        """
        Resume chunk insight generation from a previously saved pickle file.

        The pickle file contains a list of partial insight DataFrames
        produced during earlier execution. These are passed back into
        `_generate_chunk_insights`, which skips already processed chunks
        and continues extraction.

        Notes
        -----
        This mechanism enables safe recovery from interruptions such as:

        - API errors
        - process termination
        - user aborts
        """
        print("Opening pickle file to recover chunk insights generation...")
        with open(os.path.join(self.pickle_path, self.chunk_insights_pickle_file), "rb") as f:
            recover_chunk_insights = pickle.load(f)
        
        self._generate_chunk_insights(insights=recover_chunk_insights)


    def get_chunk_insights(self) -> pd.DataFrame:
        """
        Generate or recover chunk-level insights.

        This method is the public entry point for the chunk-level insight
        extraction stage. It checks whether a pickle file containing
        previously generated insights exists and prompts the user to either:

        - recover/resume the previous run, or
        - regenerate insights from scratch.

        The underlying extraction logic is implemented in
        `_generate_chunk_insights`.

        Returns
        -------
        pd.DataFrame
            Updated `corpus_state.insights` DataFrame containing chunk-level
            insights extracted from the corpus.

        Notes
        -----
        Resume functionality allows long-running LLM extraction processes
        to continue from partial results stored in a pickle file.
        """

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
        if "insight" not in self.corpus_state.insights.columns:
            raise ValueError(
                "Meta-insights cannot be created prior to generating chunk insights. "
                "Please run .get_chunk_insights before .get_meta_insights."
            )
        
        rqs = self.corpus_state.questions
        # Create the final list of paper that we will populate as we develop all the data for checking for meta insights for each paper
        list_of_papers = []
        # Get the full text and paper id
        for _, row in self.corpus_state.full_text.iterrows():
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
                paper_question_chunk_insights = self.corpus_state.insights[
                    (self.corpus_state.insights["paper_id"] == paper_id) & (self.corpus_state.insights["question_id"] == rqid)
                ]["insight"].dropna().tolist()
                paper_question_chunk_insights = "\n".join(paper_question_chunk_insights) if paper_question_chunk_insights else ""
            
                # Get the metadata for the paper id
                paper_metadata_df = self.corpus_state.insights[self.corpus_state.insights["paper_id"] == paper_id][["paper_author", "paper_title", "paper_date"]].drop_duplicates()
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

        # Update the corpus_state.chunks with the meta chunks and their ids
        temp_chunks = self.corpus_state.chunks.copy()
        # Check if the meta_chunks were already created in a previous run. If so remove them from the corpus_state object before we concat so that they don't get duplicated. 
        if temp_chunks["chunk_id"].str.startswith("meta_chunk_").any():
            temp_chunks = temp_chunks[~temp_chunks["chunk_id"].str.startswith("meta_chunk_")]
        
        self.corpus_state.chunks = pd.concat([
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
            pd.DataFrame: An updated version of corpus_state.insights which has the meta-insights appended.

        Raises:
            ValueError: If chunk insights do not exist prior to running.
        """
        # Must run chunk insights first
        if "insight" not in self.corpus_state.insights.columns:
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
                # Clean up any meta insights that have been generated in self.corpus_state.insights to avoid duplication when we generate new ones.
                self.corpus_state.insights = self.corpus_state.insights[
                    ~self.corpus_state.insights["insight_id"].astype(str).str.startswith("meta_insight_")
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
            rq_text = self.corpus_state.questions[self.corpus_state.questions["question_id"] == rq_id]["question_text"].iloc[0] if rq_id in self.corpus_state.questions["question_id"].tolist() else ""
            rq = f"{rq_id}: {rq_text}"
            other_rqs = self.corpus_state.questions[self.corpus_state.questions["question_id"] != rq_id]
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

            # Save using safe_pickle to assure atomic save
            utils.safe_pickle(meta_insights_df_lst, os.path.join(self.pickle_path, self.meta_insights_pickle_file))


        # Now join up the list of dataframes to get a single value
        meta_insights_complete_df: pd.DataFrame = pd.concat(meta_insights_df_lst).reset_index(drop=True) if meta_insights_df_lst else pd.DataFrame(columns=["paper_id", "question_id", "content_chunk_id", "meta_insight"])
        # Explode on meta_insight (currently a list for each content chunk)
        meta_insights_complete_df = meta_insights_complete_df.explode("meta_insight").reset_index(drop=True) 

        # Now we want to append to self.corpus_state.insights (which already has the chunk level insights)
        # First rename meta_insight to insight to match .corpus_state.insights, i also rename content_chunk_id to chunk_id to match the corpus_state and ensure the meta_chunks have an ID of thier own. 
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

        # Then manipulate corpus_state.insights so that we can merge with meta_insights to get a complete df that we can later append to insights
        temp_insights = self.corpus_state.insights.copy()
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
        # Now we have the meta insights with all the metadata and we can append to the corpus_state insights
        self.corpus_state.insights = pd.concat([self.corpus_state.insights, meta_insights_complete], ignore_index=True)
        # Save to parquet and return
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "08_insights"))
        return self.corpus_state.insights

    @staticmethod
    def ensure_list(x):
        """
        Normalize values into list form.

        This helper ensures that values returned from LLM outputs or
        DataFrame columns are consistently represented as lists.

        Conversion rules
        ----------------
        - list → returned unchanged
        - numpy array → converted to list
        - NA value → empty list
        - any other value → wrapped in a single-element list

        Parameters
        ----------
        x : Any
            Value to normalize.

        Returns
        -------
        list
            List representation of the input.
        """
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if pd.isna(x):
            return []
        return [x]  # fallback for any other type
    
    @staticmethod
    def estimate_tokens(text, model):
        """
        Estimate token count for a string using the model tokenizer.

        This method uses the `tiktoken` tokenizer associated with the
        specified model to estimate how many tokens a string will consume
        when sent to the LLM.

        Parameters
        ----------
        text : str
            Input text to estimate.

        model : str
            Model name used to select the appropriate tokenizer.

        Returns
        -------
        int
            Estimated token count.
        """
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @staticmethod
    def string_breaker(text, max_token_length):
        """
        Split long text into smaller segments.

        When documents exceed the model context window, this function
        breaks the text into multiple segments to allow sequential
        processing by the LLM.

        The function uses a conservative chunk size (~75% of the maximum
        token length) to avoid exceeding model limits.

        Parameters
        ----------
        text : str
            Text to split.

        max_token_length : int
            Maximum token length allowed by the model.

        Returns
        -------
        List[str]
            List of text segments.
        """
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
    
class Clustering:
    """
    Embedding and clustering stage for extracted insights.

    This class performs three computational steps on the insight corpus:

        1. Insight embedding
        2. Dimensionality reduction
        3. Density-based clustering

    These operations organize insights into provisional groups that
    provide **computational scaffolding** for downstream synthesis.

    Importantly, clusters are **not treated as analytical conclusions**.
    They serve only as an initial structural aid for theme generation,
    which later operates directly on the underlying insights and is
    refined iteratively through orphan detection and theme revision.

    Pipeline role
    -------------
    The clustering stage converts insights into a structured embedding
    representation and assigns cluster labels that allow the pipeline to:

    - summarize related insights together
    - generate initial theme schemas without scanning the full corpus
    - provide ordering heuristics for cluster summaries

    The final thematic structure is **not determined by clustering**.
    Clusters function only as a starting point for the iterative
    synthesis process.

    Attributes
    ----------
    corpus_state : CorpusState
        Working corpus state containing the insights table.

    llm_client : Any
        Client used for generating embeddings.

    embedding_model : str
        Model name used for embedding generation.

    embedding_dims : int
        Number of dimensions in the embedding vector.

    valid_embeddings_df : pd.DataFrame
        Subset of insights that contain non-empty text suitable for
        embedding generation.

    insight_embeddings_array : np.ndarray
        Full embedding vectors for valid insights.

    reduced_insight_embeddings_array : np.ndarray
        Dimensionally reduced embedding vectors produced by UMAP.

    cum_prop_cluster : pd.DataFrame
        Summary statistics describing cluster size distribution.
    """

    def __init__(
        self,
        corpus_state: CorpusState,
        llm_client: Any,
        embedding_model: str,
        embedding_dims: int = 1024,
        embeddings_pickle_path: str = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")
    ):
        """
        Initialize the clustering stage of the ReadingMachine pipeline.

        This constructor prepares the insight corpus for embedding and
        clustering operations. It validates the incoming `CorpusState`,
        removes citation parentheticals from insights to reduce embedding
        bias, and constructs the working DataFrame of valid insights that
        will be embedded.

        Parameters
        ----------
        corpus_state : CorpusState
            The current corpus state containing extracted insights and
            associated metadata.

        llm_client : Any
            Client used to generate embeddings through the embedding API.

        embedding_model : str
            Name of the embedding model used to convert insights into
            vector representations.

        embedding_dims : int, default=1024
            Dimensionality of the embedding vectors returned by the
            embedding model.

        embeddings_pickle_path : str
            Path where generated embeddings will be stored for recovery
            and reuse across runs.

        Notes
        -----
        The constructor also creates the internal working DataFrame
        `valid_embeddings_df`, which contains only insights suitable
        for embedding after citation stripping and empty-text filtering.
        """
        self.corpus_state = deepcopy(
            utils.validate_format(
            corpus_state=corpus_state,
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
        Remove citation-style parentheticals from insight text.

        Academic writing frequently embeds author-year citations
        (e.g., "(Smith 2018)" or "(Smith and Jones 2020)") that can bias
        embedding models by introducing surface-level similarities
        unrelated to semantic content.

        This function removes common citation patterns before generating
        embeddings so that clustering reflects the meaning of insights
        rather than citation artifacts.

        Parameters
        ----------
        text : str
            Raw insight text.

        Returns
        -------
        str
            Cleaned insight string with citation parentheticals removed.
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
        Prepare the subset of insights suitable for embedding.

        Insights that become empty after citation removal are excluded
        from embedding generation to prevent noise in the embedding space.

        The function also creates a normalized version of the insight text
        (`no_author_insight_string`) that is used as the embedding input.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only valid insights that will be embedded.

        Side Effects
        ------------
        Adds the column `no_author_insight_string` to
        `corpus_state.insights`.
        """

        valid = self.corpus_state.insights[
            self.corpus_state.insights["insight"].notna()
        ].copy()

        valid["no_author_insight_string"] = (
            valid["insight"]
            .astype(str)
            .apply(self._strip_citation_parentheticals)
        )

        valid = valid[valid["no_author_insight_string"].str.strip() != ""]

        # Ensure clean positional index for embedding alignment
        valid = valid.reset_index(drop=True)

        return valid
    
    def embed_insights(self) -> np.ndarray:
        """
        Generate vector embeddings for insights with incremental persistence and resume support.

        This method iterates over all valid insight strings and generates vector
        embeddings using the configured embedding model. It is designed for
        long-running jobs and provides fault tolerance through periodic checkpointing
        and resumable execution.

        Key Features
        ------------
        1. Incremental Saving
        - Embeddings are written to disk every 10 iterations using `utils.safe_pickle`.
        - This ensures progress is preserved in case of interruption (e.g., API failure,
            process termination).

        2. Resume Capability
        - If a pickle file exists, the user can choose to resume or restart.
        - On resume, previously saved embeddings are loaded and the method continues
            from the correct index (`start_idx = len(embeddings)`).
        - The DataFrame is rehydrated so completed rows already contain embeddings.

        3. Deterministic Alignment
        - Embeddings are generated in the same order as `valid_embeddings_df`.
        - Each embedding is written directly to its corresponding row using index alignment.
        - This avoids the need for joins, IDs, or post-processing merges.

        4. Data Integrity
        - `safe_pickle` ensures atomic writes, preventing file corruption.
        - Defensive handling supports edge cases such as:
            - empty or None pickle files
            - legacy numpy array formats

        Workflow
        --------
        - Initialize storage and ensure output column exists.
        - If a previous run exists:
            - Load embeddings and determine resume index.
            - Rehydrate DataFrame with completed embeddings.
        - Iterate over remaining insights:
            - Generate embedding via API call.
            - Append to in-memory list.
            - Write embedding to DataFrame at the correct index.
            - Save progress every 10 embeddings.
        - Finalize by stacking embeddings into a NumPy array and saving once more.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (n_insights, embedding_dims) containing
            embeddings for all valid insights.

        Side Effects
        ------------
        - Updates `self.valid_embeddings_df["full_insight_embedding"]` in place.
        - Updates `self.insight_embeddings_array`.
        - Writes intermediate and final embeddings to `self.embeddings_pickle_path`.

        Assumptions
        -----------
        - The order of `valid_embeddings_df` remains stable across runs.
        - The number of saved embeddings corresponds exactly to completed rows.
        - The embedding model returns consistent vector dimensions.

        Notes
        -----
        - This method replaces the need for separate `_save_embeddings` and
        `_load_embeddings` helpers.
        - Partial progress is always recoverable up to the last successful checkpoint.
        """

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.embeddings_pickle_path), exist_ok=True)

        embeddings = []
        start_idx = 0

        # Initialize column for alignment (only once)
        if "full_insight_embedding" not in self.valid_embeddings_df.columns:
            self.valid_embeddings_df["full_insight_embedding"] = None

        # --- Resume / recovery handling ---
        if os.path.exists(self.embeddings_pickle_path):
            recover = None
            while recover not in ['r', 'n']:
                recover = input(
                    f"Embeddings pickle found at '{self.embeddings_pickle_path}'. "
                    "Do you wish to resume or regenerate embeddings?\n"
                    "Hit 'r' to resume, or 'n' to regenerate (overwrite existing pickle):\n"
                ).lower()

            if recover == 'r':
                print("Loading existing embeddings for resume...")
                with open(self.embeddings_pickle_path, "rb") as f:
                    embeddings = pickle.load(f)

                # Defensive handling
                if embeddings is None:
                    embeddings = []
                elif isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()

                start_idx = len(embeddings)

                # Rehydrate into dataframe
                if start_idx > 0:
                    self.valid_embeddings_df.loc[:start_idx-1, "full_insight_embedding"] = embeddings

            else:
                print("Overwriting existing embeddings pickle...")
                embeddings = []
                start_idx = 0

        # --- Generate embeddings ---
        total = self.valid_embeddings_df.shape[0]

        for idx, insight in enumerate(
            self.valid_embeddings_df["no_author_insight_string"][start_idx:],
            start=start_idx
        ):
            print(f"Embedding insight {idx + 1} of {total}")

            response = self.llm_client.embeddings.create(
                input=insight,
                model=self.embedding_model,
                dimensions=self.embedding_dims
            )

            emb = response.data[0].embedding
            embeddings.append(emb)

            # Assign embedding to correct row
            self.valid_embeddings_df.at[idx, "full_insight_embedding"] = emb

            # Incremental save every 10 embeddings
            if (idx + 1) % 10 == 0:
                utils.safe_pickle(embeddings, self.embeddings_pickle_path)

        # --- Finalize ---
        self.insight_embeddings_array = (
            np.vstack(embeddings) if embeddings else np.array([])
        )

        # Final save to ensure completeness
        utils.safe_pickle(embeddings, self.embeddings_pickle_path)

        return self.insight_embeddings_array

    def reduce_dimensions(
        self, 
        full_embeddings: np.array = None,
        n_neighbors: int = 15,
        min_dist: float = 0.25,
        n_components: int = 10,
        metric: str = "cosine",
        random_state: int = config.seed,
        update_attributes: bool = True
    ) -> np.ndarray:
        
        """
        Reduce embedding dimensionality using UMAP.

        Dimensionality reduction improves the effectiveness of density-based
        clustering algorithms by projecting high-dimensional embeddings
        into a lower-dimensional space.

        Parameters
        ----------
        full_embeddings : np.ndarray
            Embedding matrix to reduce. Defaults to stored insight embeddings.

        n_neighbors : int
            Size of the local neighborhood used for manifold estimation.

        min_dist : float
            Minimum distance between points in the reduced space.

        n_components : int
            Number of output dimensions.

        metric : str
            Distance metric used by UMAP.

        random_state : int
            Random seed for reproducibility.

        update_attributes : bool
            Whether to update internal attributes used by downstream
            clustering methods.

        Returns
        -------
        np.ndarray
            Reduced embedding matrix.
        """
        
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
        """
        Perform density-based clustering using HDBSCAN.

        Parameters
        ----------
        embedding_matrix : np.ndarray
            Matrix of reduced embedding vectors.

        min_cluster_size : int
            Minimum cluster size parameter for HDBSCAN.

        metric : str
            Distance metric used for clustering.

        cluster_selection_method : str
            Strategy used by HDBSCAN to select clusters.

        Returns
        -------
        tuple
            cluster_labels : np.ndarray
                Cluster assignment for each insight.

            cluster_probs : np.ndarray
                Probability estimate for cluster membership.
        """
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
        """
        Compute the Davies–Bouldin score for a clustering configuration.

        The Davies–Bouldin index measures cluster compactness and
        separation. Lower scores indicate better clustering structure.

        Outlier points assigned cluster label `-1` are excluded from the
        calculation because they do not belong to any cluster.

        Parameters
        ----------
        embeddings_matrix : np.ndarray
            Matrix of embedding vectors used for clustering.

        cluster_labels : np.ndarray
            Cluster assignments produced by the clustering algorithm.

        Returns
        -------
        tuple
            db_score : float or pd.NA
                Davies–Bouldin score for the clustering configuration.
                Returns NA if fewer than two clusters exist.

            num_outliers : int
                Number of points assigned to the outlier cluster (-1).

        Notes
        -----
        The Davies–Bouldin score is used here as a diagnostic measure
        during parameter tuning rather than as a definitive evaluation
        of clustering quality.
        """


        mask = cluster_labels != -1
        num_outliers = np.sum(~mask)
        filtered_embeddings = embeddings_matrix[mask]
        filtered_labels = cluster_labels[mask]
        if len(set(filtered_labels)) < 2:
            return(pd.NA, num_outliers)
        db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
        return db_score, num_outliers

    def tune_hdbscan_params(
        self,
        min_cluster_sizes: list[int] = [5, 10, 15, 20],
        metrics: list[str] = ["euclidean", "manhattan"],
        cluster_selection_methods: list[str] = ["eom", "leaf"]
        ) -> None:

        """
        Perform grid search over HDBSCAN clustering parameters.

        This method evaluates combinations of HDBSCAN parameters across
        each research question separately. For each configuration, the
        clustering is performed and evaluated using the Davies–Bouldin
        index and the number of outlier points.

        Parameters
        ----------
        min_cluster_sizes : list[int]
            Candidate values for the HDBSCAN `min_cluster_size` parameter.

        metrics : list[str]
            Distance metrics to evaluate during clustering.

        cluster_selection_methods : list[str]
            Cluster selection strategies supported by HDBSCAN.

        Returns
        -------
        None

        Side Effects
        ------------
        Results are stored in the attribute `self.hdbscan_tuning_results`
        as a DataFrame containing the tested parameter combinations and
        their associated evaluation scores.

        Notes
        -----
        Clustering is evaluated independently for each research question
        so that insights associated with different questions do not
        influence each other's cluster structure.

        The tuning process is intended as a diagnostic tool for selecting
        reasonable clustering parameters rather than as a strict
        optimization step.
        
        """
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
        Assign clusters to insights.

        Clustering is performed separately for each research question
        so that insights addressing different questions do not influence
        each other's cluster formation.

        Cluster labels are normalized so that:

            cluster 1 = largest cluster
            cluster 2 = second largest
            ...

        Outliers are assigned label -1.

        Parameters
        ----------
        clustering_param_dict : dict
            Dictionary mapping research questions to HDBSCAN parameters.

        Returns
        -------
        pd.DataFrame
            Updated insights table containing cluster labels,
            cluster probabilities, and embedding vectors.
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

        # Clean corpus_state_insights of cluster variables that might have been created on multiple passes of generate clusters:
        cluster_cols = ["cluster", "cluster_prob", "full_insight_embedding", "reduced_insight_embedding"]
        self.corpus_state.insights = self.corpus_state.insights.drop(columns=[col for col in cluster_cols if col in self.corpus_state.insights.columns])

        self.corpus_state.insights = self.corpus_state.insights.merge(
            clustered_df[["question_id", "paper_id", "chunk_id", "insight_id", "cluster", "cluster_prob", "full_insight_embedding", "reduced_insight_embedding"]],
            on=["question_id", "paper_id", "chunk_id", "insight_id"],
            how="left"
        )

        return self.corpus_state.insights

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
    
    def clean_clusters(self, final_cluster_count: dict = None) -> pd.DataFrame:
        """
        Select the most informative clusters for each research question.

        This method retains only the largest N clusters for each research
        question and marks all remaining clusters as outliers.

        This step provides a lightweight filtering mechanism before cluster
        summarization while preserving all insights in the underlying corpus.

        Parameters
        ----------
        final_cluster_count : dict
            Mapping from `question_id` to the number of clusters to retain.

        Returns
        -------
        pd.DataFrame
            Updated insights table containing a `selected_cluster` column.
        """
        if final_cluster_count is None:
            self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "09_clusters"))
            return(self.corpus_state.insights)

        else:
            rqs = self.corpus_state.insights["question_id"].unique()
            if len(rqs) != len(final_cluster_count):
                raise ValueError(
                    "final_cluster_count must specify the number of clusters to keep for each research question."
                )

            selected_clusters_list = []

            # Loop over each research question
            for rq in self.corpus_state.insights["question_id"].unique():
                # Filter insights for the current research question
                current_rq_df = self.corpus_state.insights[self.corpus_state.insights["question_id"] == rq].copy()
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
            self.corpus_state.insights = pd.concat(selected_clusters_list)
            # Save the updated DataFrame to disk
            self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "09_clusters"))
            return self.corpus_state.insights

class Summarize:
    def __init__(self,
                 corpus_state: Any,
                 llm_client: Any,
                 ai_model: str,
                 paper_output_length: int,  # Approximate total paper length in words
                 summary_save_location: str = config.SUMMARY_SAVE_LOCATION, 
                 pickle_save_location: str = config.PICKLE_SAVE_LOCATION,
                 insight_embedding_path = os.path.join(os.getcwd(), "data", "pickles", "insight_embeddings.pkl")):
        """
        Initialize the thematic synthesis stage of the ReadingMachine pipeline.

        The `Summarize` class implements the interpretive phase of the
        ReadingMachine methodology. It transforms the structured insight
        corpus into thematic summaries through a controlled, multi-stage
        workflow coordinated with large language models.

        The synthesis pipeline implemented by this class proceeds through
        the following stages:

            1. Cluster summarization
               Summaries are generated for each insight cluster. Clusters
               are processed in semantic proximity order to improve
               contextual continuity for the model.

            2. Theme schema generation
               The model proposes a thematic schema based on cluster
               summaries. This schema defines the thematic categories and
               the rules used to assign insights to themes.

            3. Insight-to-theme mapping
               Insights are classified into one or more themes defined by
               the schema. This stage operates in batches and supports
               resumable execution.

            4. Theme population
               Each theme is populated by synthesizing the insights mapped
               to it into a narrative summary.

            5. Orphan detection and reintegration
               The system audits the synthesized summaries to ensure that
               all mapped insights are reflected in the narrative.
               Missing insights ("orphans") are reintegrated into the
               summaries through a targeted revision process.

            6. Redundancy reduction
               A final sequential pass reduces repeated information across
               themes while preserving distinct arguments.

        These steps can be iterated multiple times: theme schemas may be
        regenerated after orphan integration to refine the thematic
        structure before final synthesis.

        The class operates on two state objects:

            CorpusState
            SummaryState

        `CorpusState` contains the structured insight corpus produced by
        earlier pipeline stages (ingestion, chunking, insight extraction,
        embedding, clustering).

        `SummaryState` tracks all synthesis artifacts generated during the
        summarization process, including cluster summaries, theme schemas,
        mappings, populated themes, orphan audits, and redundancy passes.

        Initialization performs three setup steps:

            1. Verify that insight embeddings exist (produced during clustering)
            2. Load or initialize the SummaryState used to track synthesis artifacts
            3. Configure the LLM client and synthesis parameters

        Parameters
        ----------
        corpus_state : CorpusState
            Corpus state containing extracted insights and associated
            metadata produced during earlier pipeline stages.

        llm_client : Any
            Client used to call the LLM for summarization and synthesis tasks.

        ai_model : str
            Model identifier used for LLM completions.

        paper_output_length : int
            Approximate total word length for the final synthesized output.
            This value is used to proportionally allocate target lengths for
            theme summaries based on the number of insights assigned to each
            theme.

        summary_save_location : str
            Directory where summarization artifacts managed by SummaryState
            will be stored as Parquet files.

        pickle_save_location : str
            Directory used to persist intermediate artifacts during long
            summarization operations to support resumable execution.

        insight_embedding_path : str
            Path to the serialized insight embeddings generated during the
            clustering stage.

        Notes
        -----
        If existing summary artifacts are detected in `summary_save_location`,
        the user is prompted to either:

            (1) reload the existing SummaryState and resume synthesis
            (2) regenerate summaries from scratch (overwriting existing files)

        This behavior supports resumable workflows for large corpora where
        synthesis may be executed across multiple sessions.

        The summarization stage intentionally preserves intermediate
        artifacts at each step so that the evolution of the thematic
        structure can be inspected and reproduced.
        """
        
        # Check that the embeddings have been created from the clustering step. If so, load. If not send the user back to run clustering
        if not os.path.exists(insight_embedding_path):
            raise FileNotFoundError(f"Insight embeddings pickle not found at {insight_embedding_path}. Please run clustering first or amend the path to where you pickled your insight embeddings.")
        else:
            with open(insight_embedding_path, "rb") as f:
                self.insight_embeddings_array: np.ndarray = pickle.load(f)
        
        # Load the two states that this class operates on. Note this class has a SummaryState which will manage all the objects the class generates
        self.corpus_state: CorpusState = deepcopy(corpus_state)
        # Check whether SummaryState already has some summaries on disk - if so offer to reload or regenerate (regen will overwrite everything)
        os.makedirs(summary_save_location, exist_ok=True)
        path = Path(summary_save_location)
        possible_summary_files = list(path.glob("*.parquet"))
        if possible_summary_files:
            load = None
            while load not in ["1", "2"]:
                load = input(
                    f"Previous summarization files found in '{summary_save_location}'. Do you want to:\n"
                    "(1) reload existing summaries\n"
                    "(2) regenerate summaries (NOTE: this will overwrite all existing summary attributes)\n"
                    "Enter 1 or 2:\n"
                ).strip()

            if load == "1":
                self.summary_state = SummaryState.load(summary_save_location=summary_save_location)
                print("Summaries loaded. To see where you are in the summarization process run var.summary_state.status())")

            else:
                # Clear out existing summaries if we are regenerating
                for file in Path(summary_save_location).glob("*.parquet"):
                    file.unlink()  
                # And load the empty summary_state
                self.summary_state = SummaryState(summary_save_location=summary_save_location)
        
        # If no summary files found, just load the empty summary state
        else:
            self.summary_state = SummaryState(summary_save_location=summary_save_location)

        self.llm_client: Any = llm_client
        self.ai_model: str = ai_model
        self.paper_output_length: int = paper_output_length
        self.summary_save_location = summary_save_location
        pickle_save_location = pickle_save_location

    def _calculate_centroid(self, col="full_insight_embedding"):
        """
        Compute cluster centroids for each research question.

        For each `(question_id, cluster)` pair, this method aggregates the
        embedding vectors associated with the cluster and computes their
        mean vector (centroid). These centroids are later used to estimate
        an ordering of clusters based on semantic proximity.

        Parameters
        ----------
        col : str, default="full_insight_embedding"
            Column in `corpus_state.insights` containing the embedding
            vectors used to compute centroids.

        Returns
        -------
        pd.DataFrame
            DataFrame containing centroid vectors for each cluster with
            columns:

                - question_id
                - cluster
                - centroid

        Notes
        -----
        Several safeguards are applied during centroid calculation:

        - Insights without valid embeddings are skipped.
        - Ragged embedding vectors (inconsistent lengths) are filtered
        using the modal vector length.
        - Embeddings containing NaN values are excluded.

        These checks ensure centroid computation remains stable even if
        some embeddings are malformed or missing.
        """
        rows = []
        for rq, d in self.corpus_state.insights.groupby("question_id", sort=False):
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
        """
        Estimate an ordering of clusters based on centroid similarity.

        This method computes centroids for each cluster and then determines
        an approximate shortest path through those clusters using pairwise
        cosine distances between centroid embeddings.

        The resulting ordering places semantically similar clusters next
        to one another. This ordering is later used when generating cluster
        summaries so that related clusters appear in adjacent positions,
        providing the model with coherent contextual scaffolding during
        theme generation.

        Procedure
        ---------
        1. Compute cluster centroids from insight embeddings.
        2. Compute pairwise cosine distances between centroids.
        3. Estimate a shortest path through the clusters:

        - For small cluster counts (<10), all permutations are evaluated
            to find the optimal path.
        - For larger cluster sets, an approximate Traveling Salesman
            solution from NetworkX is used.

        4. Append the outlier cluster (-1) at the end of the ordering.

        Returns
        -------
        dict
            Dictionary mapping each research question to an ordered list
            of clusters:

                {
                    question_id: {
                        "order": [cluster_1, cluster_2, ..., -1]
                    }
                }

        Notes
        -----
        This ordering step does not influence the final thematic structure.
        It is used only to provide a coherent sequence of cluster summaries
        during the initial theme generation stage.
        """
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
                final_order = order + [-1]
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
    

    def summarize_clusters(self):
        """
        Generate summaries for each cluster of insights across all research questions.

        This method performs the first stage of the thematic synthesis process by
        summarizing the insights contained within each cluster. Cluster summaries
        provide an initial structural overview of the corpus and act as scaffolding
        for the subsequent theme generation stage.

        The clusters are processed in the order determined by the centroid shortest
        path algorithm (`_estimate_shortest_path`). This ordering places semantically
        similar clusters adjacent to each other so that the model receives coherent
        contextual information when generating summaries.

        During summarization, previously generated cluster summaries are passed to
        the model as frozen context. This allows the model to maintain consistency
        across summaries while preventing earlier summaries from being modified.

        If cluster summaries already exist on disk, the user is prompted to either:

            (1) reload existing summaries
            (2) regenerate summaries

        Regenerating summaries will reset the entire summarization pipeline because
        all downstream artifacts depend on cluster summaries. Specifically, the
        following artifacts are cleared if regeneration is selected:

            - theme schemas
            - insight-to-theme mappings
            - populated theme summaries
            - orphan detection results
            - redundancy pass outputs

        Returns
        -------
        List[pd.DataFrame]
            A list containing a single DataFrame with the generated cluster
            summaries. The DataFrame includes the following columns:

                - question_id
                - question_text
                - cluster
                - summary

        Notes
        -----
        Cluster summaries are not treated as analytical conclusions. They serve
        only as a structural aid for theme generation, which subsequently operates
        directly on the underlying insights and is refined iteratively through
        orphan detection and schema updates.

        The generated summaries are stored in `self.summary_state.cluster_summary_list`
        and persisted to disk via `SummaryState.save()` to support resumable
        workflows.
        """
        
        if self.summary_state.cluster_summary_list is not None and len(self.summary_state.cluster_summary_list) > 0:
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
                    "Summaries can be accessed via the variable.cluster_summary_list attribute of this class. There is only one item in the list thus: `variable.cluster_summary_list[0]`."
                    )
                return(None) # Return to exit the function and avoid re-running summarization
            else:
                # If we are regenerating summeries we need to delete all existing outputs and reset the attributes to none as they are loaed on init if they exist
                print("Re-running summarization of clusters...")
                # Delete all the existing outputs and set thier values to empty lists
                self.summary_state.restart(confirm="yes")
    
        # We are going to send the insights to the LLM in the order of the shortest path, so that the most similar clusters are summarized close together
        # This will add coherence to the final paper when the summaries are stitched together
        # It will also aid in the applicaion of the sliding window for summary clean up
        shortest_paths = self._estimate_shortest_path()
        
        # Create list to populate with summaries from the LLM
        summaries_dict_lst: List[dict] = []

        # Get the numbers to show progress 
        total_clusters = sum(len(shortest_paths[rq_id]["order"]) for rq_id in shortest_paths)
        count = 1

        # Loop over unique research questions
        for _, row in self.corpus_state.questions.iterrows():
            rq_id = row["question_id"]
            rq_text = row["question_text"]
            rq_df: pd.DataFrame = self.corpus_state.insights[self.corpus_state.insights["question_id"] == rq_id].copy()

            # Loop over clusters for this research question - in shortest path order
            for cluster in shortest_paths[rq_id]["order"]:
                print(f"Summarizing cluster {cluster} for research question {rq_id} (total progress: {count} of {total_clusters})...")
                count += 1
                # Skip any cases where chunks might have had no insights (and therefore no cluster)
                if pd.isna(cluster) or cluster == "NA":
                    continue

                cluster_df: pd.DataFrame = rq_df[rq_df["cluster"] == cluster]
                # get the insights, they are list of single strings. So make sure they are valid string to send to the LLM
                insights: List[str] = cluster_df["insight"].apply(
                    lambda x: x if isinstance(x, str) else (
                        x[0] if isinstance(x, list) and len(x) == 1 and isinstance(x[0], str) else None
                    )).tolist()
                
                if any(i is None for i in insights):
                    raise ValueError("Insight format error: each insight must be a string or a single-item list containing a string.")

                # Build system prompt from predefined method
                sys_prompt: str = Prompts().summarize_clusters()

                # Get the summaries frozen so far if there are any
                frozen_summaries = pd.DataFrame(summaries_dict_lst)["summary"].tolist() if summaries_dict_lst else []

                frozen_summaries_str = "\n".join(frozen_summaries) if frozen_summaries else ""
                insights_str = "\n".join(insights)

                # Build user prompt for LLM
                user_prompt: str = (
                    f"Research question id: {rq_id}\n"
                    f"Research question text: {rq_text}\n"
                    "PRECEDING CLUSTER SUMMARIES (for context only; may be empty):\n"
                    f"{frozen_summaries_str}\n"
                    f"Cluster: {cluster}\n"
                    "INSIGHTS:\n" 
                    f"{insights_str}\n"
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

        self.summary_state.cluster_summary_list = [summaries_df]

        # Save summaries as this is LLM output we may want to reuse later - save as parquet
        os.makedirs(self.summary_save_location, exist_ok=True)
        self.summary_state.save()

        return self.summary_state.cluster_summary_list
    
    def _run_llm_schema_gen(self, source: str) -> pd.DataFrame:
        """
        Generate a theme schema using the LLM.

        This internal helper constructs the input data for schema generation
        and performs the LLM call that produces a set of candidate themes
        for each research question.

        The method supports two input sources:

            - "cluster summaries": used for the first schema pass
            - "populated themes": used during iterative refinement passes

        In the first pass, cluster summaries act as scaffolding for theme
        generation. In later passes, the model receives the current thematic
        summaries (including orphan reinsertion effects) to refine the
        schema structure.

        Parameters
        ----------
        source : str
            Input data source used for schema generation. Must be one of:

                "cluster summaries"
                "populated themes"

        Returns
        -------
        pd.DataFrame
            DataFrame containing the generated theme schema. Columns include:

                - theme_id
                - theme_label
                - theme_description
                - instructions
                - question_id
                - question_text

        Notes
        -----
        The LLM returns theme definitions consisting of:

            - theme_label: short name for the theme
            - theme_description: description of the thematic category
            - instructions: rules for assigning insights to the theme

        A numeric `theme_id` is assigned after generation to preserve a
        stable ordering of themes during later synthesis stages.
        
        """
        if source not in ["cluster summaries", "populated themes"]:
            raise ValueError("Invalid source for theme schema generation. Source must be either 'cluster summaries' or 'populated themes'.")
        
        if source == "cluster summaries":
            # Grab data from summaries
            source_df = self.summary_state.cluster_summary_list[0].copy()
            source_df.rename(columns={"cluster": "id", "summary": "text_to_theme"}, inplace=True)
        else:
            # if its an iteration we get data from the last populated theme and convert the columsn to a generic form to send to the llm
            source_df = self.summary_state.populated_theme_list[-1].copy()
            source_df.rename(columns={"theme_id": "id", "thematic_summary": "text_to_theme"}, inplace=True)
        
        out_df_list = []

        for idx, row in self.corpus_state.questions.iterrows():
            print(f"Generating theme schema for question {row['question_id']} (total: {idx + 1} of {len(self.corpus_state.questions)})...")
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
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "theme_label": { "type": "string" },
                                    "theme_description": { "type": "string" },
                                    "instructions": { "type": "string" }
                                },
                                "required": [
                                    "theme_label", 
                                    "theme_description", 
                                    "instructions"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["themes"],
                    "additionalProperties": False
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
            themes_df = pd.DataFrame(theme_list, columns=["theme_label", "theme_description", "instructions"])
            themes_df["question_id"] = question_id
            themes_df["question_text"] = question_text

            out_df_list.append(themes_df)
        
        # Concat all the questions
        output = pd.concat(out_df_list, ignore_index=True, sort=False)
        # Add a numeric id to the themes so that i can sort them later which is important to generate the narrative at the end.
        # NOTE THIS IS A CENTRAL CONDITION. FOR THIS REASON THERE IS A FLAGGING FUNCTION AT LOAD AND SAVE WHICH COMPLAINS TO THE USER IF SOME CHANGE TO THE CODE HAS RESULTED IN theme_id NOT BEING AN INT.
        output["theme_id"] = [i + 1 for i in range(len(output))] 

        return(output)

    def gen_theme_schema(self, force: bool = False) -> pd.DataFrame:
        
        """
        Generate or update the theme schema for the synthesis process.

        This method manages the iterative theme schema generation stage
        of the pipeline. The schema defines the thematic categories that
        will be used to organize insights during synthesis.

        Schema generation operates in three modes:

            1. Initial pass
            If no schema exists, themes are generated from cluster
            summaries.

            2. Iterative pass
            After insights have been mapped, themes populated, and
            orphans handled, a new schema can be generated from the
            updated thematic summaries.

            3. Regeneration
            The most recent schema pass can be overwritten and
            regenerated.

        Parameters
        ----------
        force : bool, default=False
            If True, bypass sequencing validation and generate a new
            schema directly from cluster summaries.

            This mode is intended for development or testing and may
            leave the pipeline state inconsistent.

        Returns
        -------
        pd.DataFrame
            The newly generated theme schema.

        Raises
        ------
        ValueError
            If required upstream stages have not been completed or if
            schema sequencing rules are violated.

        Notes
        -----
        Theme schemas are stored as sequential passes in
        `self.summary_state.theme_schema_list`.

        Each new schema pass reflects the evolving thematic structure
        after incorporating orphan insights and updated theme summaries.
        """
        
        if force not in [True, False]:
            raise ValueError("Invalid value for 'force' parameter. Must be a boolean (True or False).")

         # --- Root validation ---
        if not self.summary_state.cluster_summary_list:
            raise ValueError(
                "No cluster summaries found. Please run cluster summarization first."
            )
        ##### 
        # Experimental mode where force is True and guardrails are skipped
        if force:
            print("WARNING: Force flag is True: Skipping validation and sequencing checks and generating new schema pass from cluster summaries. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            new_schema = self._run_llm_schema_gen(source="cluster summaries")
            self.summary_state.theme_schema_list.append(new_schema)
            self.summary_state.save()
            return new_schema
        #####

        # Else run all the regular validation and sequencing checks and proceed as the user specifies/system alllows            
        schema_len = len(self.summary_state.theme_schema_list)
        populate_len = len(self.summary_state.populated_theme_list)
        orphan_len = len(self.summary_state.orphan_list)

        # Determine the input source (from cluster summaries or from orphan handling):
        # --- CASE 1: First schema pass, build from clusters
        if schema_len == 0:
            source_name = "cluster summaries"
            new_schema = self._run_llm_schema_gen(source=source_name)
            self.summary_state.theme_schema_list.append(new_schema)
            self.summary_state.save()
            return new_schema  

        # --- Existing schema passes, check what user wants---
        choice = None
        while choice not in ["1", "2", "3"]:
            choice = input(
                f"You currently have {schema_len} theme schema pass(es).\n"
                "Choose an option:\n"
                "(1) Load the latest schema pass\n"
                "(2) Generate a NEW schema pass (requires orphan incorporation)\n"
                "(3) Regenerate the LATEST schema pass (overwrite last pass)\n"
                "Enter 1, 2, or 3:\n"
            ).strip()

        # --- Option 1: Reload ---
        if choice == "1":
            print("Latest schema loaded.")
            return None

        # --- Option 2: Append new iteration ---
        if choice == "2":

            # Enforce full structural cycle completion
            if populate_len == 0:
                raise ValueError(
                    "No populated themes found. Please run populate_themes() first."
                )

            if populate_len != orphan_len:
                raise ValueError(
                    "You must run handle_orphans() before generating a new schema pass."
                )

            source_name = "populated themes"
            new_schema = self._run_llm_schema_gen(source=source_name)
            self.summary_state.theme_schema_list.append(new_schema)

            self.summary_state.save()
            return new_schema

        # --- Option 3: Regenerate last pass ---
        if choice == "3":

            # Rewind to just before the last schema pass
            last_index = schema_len - 1
            self.summary_state.rewind_to("schema", last_index)

            # Re-evaluate state AFTER rewind
            populate_len_after = len(self.summary_state.populated_theme_list)

            if last_index == 0 and populate_len_after == 0:
                source_name = "cluster summaries"
            else:
                source_name = "populated themes"

            new_schema = self._run_llm_schema_gen(source=source_name)

            # Replace last schema pass
            self.summary_state.theme_schema_list[-1] = new_schema

            self.summary_state.save()
            return new_schema

    def _validate_and_cast_theme_ids(self, df, allowed_ids):
        """
        Validate theme identifiers returned by the LLM.

        During insight-to-theme mapping, the LLM returns theme IDs that
        correspond to themes defined in the schema. This method verifies
        that all returned IDs are valid and converts them to integer type.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing theme assignments returned by the LLM.
            Must include a `theme_id` column.

        allowed_ids : Iterable
            Collection of valid theme identifiers defined by the current
            theme schema.

        Returns
        -------
        pd.DataFrame
            DataFrame with validated and type-cast `theme_id` values.

        Raises
        ------
        ValueError
            If the LLM returns any theme identifiers not present in the
            allowed set.
        """
        allowed_set = set(str(i) for i in allowed_ids)
        returned_set = set(df["theme_id"].astype(str))

        invalid_ids = returned_set - allowed_set
        if invalid_ids:
            raise ValueError(
                f"Invalid theme_id(s) returned by LLM: {invalid_ids}. "
                f"Allowed IDs: {sorted(allowed_set)}"
            )

        df["theme_id"] = df["theme_id"].astype(int)
        return df
    

    def _map_insights_via_llm(
        self, 
        batch_size,
        already_mapped_insight_ids,
        mapped_insights_df_list, 
        in_progress_path, 
        mode
        ):
        """
        Map insights to themes using the LLM.

        This internal method performs the core insight-to-theme classification
        stage of the pipeline. Insights are processed in batches and assigned
        to one or more themes defined in the current theme schema.

        Each batch of mappings is saved to disk during execution to support
        resumable operation and prevent loss of progress during long runs.

        Parameters
        ----------
        batch_size : int
            Number of insights to process per LLM call.

        already_mapped_insight_ids : list
            List of insight IDs that have already been mapped. Used during
            resume operations to avoid reprocessing previously completed
            batches.

        mapped_insights_df_list : list
            List of DataFrames containing previously generated mapping
            results. New batches are appended to this list.

        in_progress_path : str
            File path where intermediate mapping progress is serialized
            during execution.

        mode : str
            Execution mode controlling validation behavior. Must be one of:

                "normal" — standard execution with state validation
                "force"  — bypasses state integrity safeguards

        Returns
        -------
        pd.DataFrame
            DataFrame containing the complete set of mapped insights with
            the following columns:

                - insight_id
                - theme_id
                - question_id

        Notes
        -----
        A state fingerprint is stored alongside intermediate results to
        ensure that the corpus and summary states have not changed between
        resume attempts. If a mismatch is detected, resume is aborted to
        prevent corruption of the synthesis pipeline.
        """
        
        if mode not in ["force", "normal"]:
            raise ValueError("Invalid mode. Mode must be either 'force' or 'normal'.")
        
        # Create a meta object of the state against which any resume can be checked to ensure state has not been changed between resume calls
        # This gets pickled along with the progess in mapped insights
        state_meta = {
            "corpus_hash": self.corpus_state.fingerprint(),
            "summary_hash": self.summary_state.fingerprint()
        }

        # Work on a temporary copy
        temp_state_insights = self.corpus_state.insights.copy()

        # Remove already mapped insights if resuming
        if already_mapped_insight_ids:
            temp_state_insights = temp_state_insights[
                ~temp_state_insights["insight_id"].isin(
                    already_mapped_insight_ids
                )
            ]

        # Iterate through each research question
        for _, q_row in self.corpus_state.questions.iterrows():

            question_id = q_row["question_id"]
            question_text = q_row["question_text"]

            # Filter insights for this question
            q_insights_df = temp_state_insights[
                (temp_state_insights["question_id"] == question_id) &
                (temp_state_insights["insight"].notna()) &
                (temp_state_insights["insight"] != "")
            ].copy()

            if q_insights_df.empty:
                continue

            # Get schema for this question
            q_schema_df = self.summary_state.theme_schema_list[-1][
                self.summary_state.theme_schema_list[-1]["question_id"] == question_id
            ].copy()

            q_schema_json = q_schema_df[
                ["theme_id", "theme_label", "theme_description", "instructions"]
            ].to_json(orient="records", indent=2)

            # Pre-build JSON schema (explicit)
            json_schema = {
                "name": "insight_to_theme_mapper",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "mapped_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "insight_id": {"type": "string"},
                                    "theme_id": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["insight_id", "theme_id"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["mapped_data"],
                    "additionalProperties": False
                }
            }

            # Calculate total batches for this question
            total_batches_for_question = math.ceil(
                len(q_insights_df) / batch_size
            )

            # Batch loop
            for batch_index, start_index in enumerate(
                range(0, len(q_insights_df), batch_size),
                start=1
            ):

                print(
                    f"Mapping insights to themes for question "
                    f"{question_id} - batch "
                    f"{batch_index} of "
                    f"{total_batches_for_question}..."
                )

                batch_df = q_insights_df.iloc[
                    start_index : start_index + batch_size
                ]

                if batch_df.empty:
                    continue

                current_batch_str = "\n".join(
                    f"{row.insight_id}: {row.insight}"
                    for row in batch_df.itertuples()
                )

                # Get the inputs to the sys prompt call: the list of allowd theme_ids to tag to, the other theme id and the conflict theme id
                allowed_theme_ids = q_schema_df["theme_id"].astype(int).tolist()
                other_theme_rows = q_schema_df[q_schema_df["theme_label"].str.lower() == "other"]["theme_id"].astype(str)
                other_theme_id = other_theme_rows.iloc[0] if not other_theme_rows.empty else None
                conflicts_theme_rows = q_schema_df[q_schema_df["theme_label"].str.lower() == "conflict"]["theme_id"].astype(str)
                conflicts_theme_id = conflicts_theme_rows.iloc[0] if not conflicts_theme_rows.empty else None

                sys_prompt = Prompts().theme_map_to_schema(
                    allowed_ids=allowed_theme_ids,
                    other_theme_id=other_theme_id,
                    conflicts_theme_id=conflicts_theme_id
                    )
                
                user_prompt = (
                    f"RESEARCH QUESTION: {question_text}\n"
                    "THEMATIC CODEBOOK:\n"
                    f"{q_schema_json}\n\n"
                    "INSIGHTS TO MAP:\n"
                    f"{current_batch_str}\n\n"
                )

                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    fall_back={"mapped_data": []},
                    return_json=True,
                    json_schema=json_schema
                )

                batch_results_df = pd.DataFrame(
                    response.get("mapped_data", [])
                )

                if batch_results_df.empty:
                    continue

                # Expand theme_id
                batch_results_df = batch_results_df.explode("theme_id")
                batch_results_df["question_id"] = question_id
                # Validate that all returned theme_ids are valid
                batch_results_df = self._validate_and_cast_theme_ids(
                    batch_results_df,
                    allowed_theme_ids
                )

                mapped_insights_df_list.append(batch_results_df)
                mapped_insights_df_with_meta = {"mapped_insights_df_list": mapped_insights_df_list, 
                                                "state_meta": state_meta, 
                                                "mode": mode}

                # Save progress safely after each batch = use safe pickle to avoid corruption if something goes wrong during the write
                utils.safe_pickle(mapped_insights_df_with_meta, in_progress_path)
            
        # Check insights existed and were mapped, then concat and return the mapped insights
        if not mapped_insights_df_list:
            mapped_insights_df = pd.DataFrame()
        else:
            mapped_insights_df = pd.concat(
                mapped_insights_df_list,
                ignore_index=True
            )

        return(mapped_insights_df)
    
    def map_insights_to_themes(
        self,                       
        batch_size=75, 
        force = False
        )-> pd.DataFrame:
        """
        Map all insights to themes defined in the current theme schema.

        This method is the public entry point for the insight-to-theme
        mapping stage of the synthesis pipeline. It orchestrates batch
        classification of insights using the LLM and manages resume logic,
        state validation, and persistence of mapping results.

        Parameters
        ----------
        batch_size : int, default=75
            Number of insights sent to the LLM in each batch.

        force : bool, default=False
            If True, bypass state validation and resume checks and perform
            a fresh mapping run. This mode is intended for development and
            testing and may leave the pipeline state inconsistent.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the mapping of insights to themes for the
            most recent pass.

        Raises
        ------
        ValueError
            If no theme schema exists or if a resume attempt detects that
            the underlying corpus or summary state has changed.

        Workflow
        --------
        The method proceeds through the following stages:

            1. Validate that a theme schema exists.
            2. Detect whether a partial mapping process is available for
            resumption.
            3. If resuming, verify that the corpus and summary states match
            the state recorded in the progress file.
            4. Perform batch mapping of insights using the LLM.
            5. Persist mapping results in `SummaryState.mapped_theme_list`.
            6. Save the updated summary state and remove any temporary
            progress files.

        Notes
        -----
        Insight-to-theme mapping is performed independently for each
        research question and is constrained by the current theme schema.

        Each insight may be assigned to multiple themes if appropriate,
        including special categories such as "Other" or "Conflict".
        """

        # Check that it is possible to run this step: i.e. there is a theme schema to map to
        if not self.summary_state.theme_schema_list:
            raise ValueError(
                "No theme schema found. Please run gen_theme_schema() first. No force option possible."
            )
        
        # Set up the variabales i will need in the logic below
        # Get the lengths of the theme schema and mapped themes to handle state integirity
        schema_len = len(self.summary_state.theme_schema_list)
        mapped_len = len(self.summary_state.mapped_theme_list)

        # First check for resume 
        #### --------------------------------------------------
        #### RESUMPTION / RECOVERY LOGIC
        #### --------------------------------------------------

        # Get the progress path for the mapping process as i will need this immediately in the logic
        in_progress_path = os.path.join(
            self.summary_save_location, "mapped_theme_in_progress.pickle"
        )

        # Set this as empty for no resume and update the value in the resume branch if triggered
        already_mapped_insight_ids = []

        # Check for existing in-progress file
        if os.path.exists(in_progress_path):

            resume_choice = None
            while resume_choice not in ["1", "2"]:
                resume_choice = input(
                    "A partial mapping process was detected. Do you want to:\n"
                    "1) resume from the last saved point? \n"
                    "2) start a new mapping process? \n"
                    "Enter 1 or 2:\n"
                ).strip()

            if resume_choice == "1":
                with open(in_progress_path, "rb") as f:
                    mapped_insights_df_list_meta = pickle.load(f)
                    mapped_insights_df_list = mapped_insights_df_list_meta["mapped_insights_df_list"]
                    pickled_meta = mapped_insights_df_list_meta["state_meta"]
                    mode = mapped_insights_df_list_meta["mode"]
                
                # Check that the state has not changed between resume calls
                current_meta = {
                    "corpus_hash": self.corpus_state.fingerprint(),
                    "summary_hash": self.summary_state.fingerprint()
                }

                if current_meta != pickled_meta:
                    raise ValueError(
                        "State change detected between resume call. Resume invalid to prevent state corruption. "
                        "Force override if you know what you are doing or run without resume."
                    )

                if mapped_insights_df_list:
                    recovered_df = pd.concat(
                        mapped_insights_df_list,
                        ignore_index=True
                    )
                    already_mapped_insight_ids = (
                        recovered_df["insight_id"].unique().tolist()
                    )

                print("Resuming mapping process from last saved point...")
                mapped_insights_df = self._map_insights_via_llm(
                    batch_size=batch_size,
                    already_mapped_insight_ids=already_mapped_insight_ids,
                    mapped_insights_df_list=mapped_insights_df_list,
                    in_progress_path=in_progress_path, 
                    mode = mode
                )

                # Now check the mode to know how to handle.
                # If the mode is force we append as that is the defualt for force
                if mode == "force":
                    self.summary_state.mapped_theme_list.append(mapped_insights_df)
                    # Save state
                    os.makedirs(self.summary_save_location, exist_ok=True)
                    self.summary_state.save()
                    # Remove in-progress file now that mapping and save is complete
                    if os.path.exists(in_progress_path):
                        os.remove(in_progress_path)
                    # Return to exit the function
                    return self.summary_state.mapped_theme_list[-1]
                # mode is normal, so we follow pattern for ensuring state integrity
                else: 
                    # Make sure the state is integreal by realigning the schema and mapping history to the last coherent point 
                    min_len = min(schema_len, mapped_len)
                    self.summary_state.theme_schema_list = self.summary_state.theme_schema_list[:min_len + 1]
                    self.summary_state.mapped_theme_list = self.summary_state.mapped_theme_list[:min_len]
                    # Append the new mapped insights to the state
                    self.summary_state.mapped_theme_list.append(mapped_insights_df)
                    # Save state
                    os.makedirs(self.summary_save_location, exist_ok=True)
                    self.summary_state.save()
                    # Remove in-progress file now that mapping and save is complete
                    if os.path.exists(in_progress_path):
                        os.remove(in_progress_path)
                    # Return to exit the function
                    return self.summary_state.mapped_theme_list[-1]
            else:
                os.remove(in_progress_path)
                print("Deleted in-progress file. Starting new mapping process.")


        ####--------------------------------------------------
        #  Force flag enabled - skip all validation and resume checks. Run a mapping of the insights to the latest themes and append to the state.
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and mapping insights to themes. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            
            # Get the mapped df from the llm with resume elements set to empty as we want fresh run
            mapped_insights_df = self._map_insights_via_llm(
                batch_size=batch_size,
                already_mapped_insight_ids=[],
                mapped_insights_df_list=[],
                in_progress_path=in_progress_path, 
                mode = "force"
            )
            
            # Append to state (because force defaults to append), save, clean-up and return
            self.summary_state.mapped_theme_list.append(mapped_insights_df)
            self.summary_state.save()
            # Remove in-progress file now that mapping and save is complete
            if os.path.exists(in_progress_path):
                os.remove(in_progress_path)
            return mapped_insights_df
        ######-----------------------------

        # If mappings exist, offer choice
        if mapped_len > 0:
            new_choice = None
            while new_choice not in ["1", "2"]:
                new_choice = input(
                    "Mapped themes already exist on disk. Do you want to:\n"
                    "(1) reload existing mapped themes\n"
                    "(2) remap insights to the latest theme schema. "
                    "This will realign schema and mapping history.\n"
                    "Enter 1 or 2:\n"
                ).strip()

            if new_choice == "1":
                print("Mapped themes loaded.")
                return None

            # Realign to last coherent pre-mapping corpus_state
            min_len = min(schema_len, mapped_len)
            self.summary_state.theme_schema_list = self.summary_state.theme_schema_list[:min_len + 1]
            self.summary_state.mapped_theme_list = self.summary_state.mapped_theme_list[:min_len]

            # Get the mappings from the llm - passsing empty values for the resume elements as this is a fresh run
            mapped_insights_df = self._map_insights_via_llm(
                batch_size=batch_size,
                already_mapped_insight_ids=[],
                mapped_insights_df_list=[],
                in_progress_path=in_progress_path, 
                mode = "normal"
            )

        else:
            # No mappings exist — ensure we are in valid pre-mapping corpus_state
            # This is duplicative because we tested earlier, but this makes the logic more explicit
            if schema_len < 1:
                raise ValueError("No schema available to map.")
            
            # If no mappings exist and there is no resume and as we are not forcing, then we should run under normal mode (i.e. a fresh run)
            mapped_insights_df = self._map_insights_via_llm(
                batch_size=batch_size,
                already_mapped_insight_ids=[],
                mapped_insights_df_list=[],
                in_progress_path=in_progress_path,
                mode="normal"
            )
        
        #### --------------------------------------------------
        #### FINALIZATION ON NORMAL MODE (non-force)
        #### --------------------------------------------------

        # Update state
        self.summary_state.mapped_theme_list.append(mapped_insights_df)
        # Save state
        os.makedirs(self.summary_save_location, exist_ok=True)
        self.summary_state.save()
        # Remove in-progress file now that mapping and save is complete
        if os.path.exists(in_progress_path):
            os.remove(in_progress_path)

        return self.summary_state.mapped_theme_list[-1]


    def _estimate_theme_lengths(self, paper_len: int, max_model_output_words: int = 2800) -> pd.DataFrame:
        """
        Estimate target word lengths for each theme.

        This method allocates a target word length to each theme based on the
        number of insights mapped to that theme relative to the total insight
        count across the research question.

        The allocation process follows three steps:

            1. Count insights mapped to each theme.
            2. Compute each theme's proportion of the total insights.
            3. Allocate word length proportionally based on the desired total
            paper length.

        Hard bounds are applied to prevent extreme allocations:

            - Minimum theme length: 375 words
            - Maximum theme length: model output ceiling

        Parameters
        ----------
        paper_len : int
            Approximate total desired length of the final synthesis document
            in words.

        max_model_output_words : int, default=2800
            Maximum number of words the model can reliably produce in a single
            response. This acts as the upper bound for theme summaries.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:

                - theme_id
                - allocated_length

        Notes
        -----
        Themes that have zero mapped insights are preserved and assigned the
        minimum length. This maintains schema stability across synthesis passes
        even when some themes temporarily receive no insights.
        """

        MIN_THEME_WORDS = 375
        MAX_THEME_WORDS = max_model_output_words

        # --- Full list of expected themes (anchor) ---
        full_schema = (
            self.summary_state.theme_schema_list[-1][["theme_id"]]
            .copy()
            .astype({"theme_id": int})
            .drop_duplicates()
        )

        # --- Count mapped insights per theme ---
        theme_counts = (
            self.summary_state.mapped_theme_list[-1]
            .copy()
            .astype({"theme_id": int})
            .groupby("theme_id")
            .size()
            .reset_index(name="count")
        )

        # --- Merge to preserve zero-hit themes ---
        theme_counts = full_schema.merge(
            theme_counts,
            on="theme_id",
            how="left"
        )

        theme_counts["count"] = theme_counts["count"].fillna(0)

        # --- Compute global proportions ---
        total_count = theme_counts["count"].sum()

        if total_count > 0:
            theme_counts["proportion"] = (
                theme_counts["count"] / total_count
            )
        else:
            theme_counts["proportion"] = 0

        # --- Allocate proportionally ---
        theme_counts["allocated_length"] = np.ceil(
            theme_counts["proportion"] * paper_len
        ).astype(int)

        # --- Apply hard floor and hard ceiling ---
        theme_counts["allocated_length"] = (
            theme_counts["allocated_length"]
            .clip(lower=MIN_THEME_WORDS, upper=MAX_THEME_WORDS)
        )

        # Defensively ensure that theme_id remains an int to allow for sorting
        theme_counts["theme_id"] = theme_counts["theme_id"].astype(int)

        return theme_counts[["theme_id", "allocated_length"]]


    def _check_length_and_flag(self, df: pd.DataFrame, max_prop: float, max_model_output_words: int = 2800) -> pd.DataFrame:
        """
        Identify themes whose summaries exceed a specified proportion of
        their allocated length.

        This function detects cases where the model has compressed a theme
        excessively in order to fit within the allocated word limit. When
        the current summary length exceeds the specified proportion of the
        allocated length, the theme is flagged for potential expansion.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of populated themes containing:

                - allocated_length
                - current_length

        max_prop : float
            Proportion of allocated length beyond which the summary will be
            flagged for expansion.

        max_model_output_words : int, default=2800
            Maximum output length supported by the model. Themes already at
            this ceiling are not flagged.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Two DataFrames:

                df_len_ok
                    Themes within acceptable length limits.

                df_len_flagged
                    Themes exceeding the specified proportion of their
                    allocated length.
        """

        MAX_THEME_WORDS = max_model_output_words

        def flag_row(row):
            # Treat missing summaries as not flagged
            if pd.isna(row["current_length"]):
                return 0

            # Do not flag if theme is already at hard ceiling
            if row["allocated_length"] == MAX_THEME_WORDS:
                return 0
            
            # Standard proportional check
            if row["current_length"] <= row["allocated_length"] * max_prop:
                return 0
            else:
                return 1

        df["length_flag"] = df.apply(flag_row, axis=1)

        df_len_ok = df[df["length_flag"] == 0]
        df_len_flagged = df[df["length_flag"] == 1]

        return df_len_ok, df_len_flagged
    
    def _run_theme_pop(
        self, 
        schema_df: pd.DataFrame,
        mapped_themes_df: pd.DataFrame, 
        paper_len = 8000
        ) -> pd.DataFrame:

        """
        Populate themes by synthesizing insights assigned to each theme.

        This method generates narrative summaries for each theme by
        synthesizing the insights mapped to that theme. Each theme is
        processed independently using the LLM.

        The function:

            1. Estimates target theme lengths if not already present.
            2. Retrieves the insights mapped to each theme.
            3. Builds prompts tailored to the theme type.
            4. Calls the LLM to synthesize a thematic summary.

        Special theme types are handled differently:

            - "conflicts": emphasizes incompatible claims
            - "other": captures minority insights
            - "general": standard synthesis

        Parameters
        ----------
        schema_df : pd.DataFrame
            Current theme schema containing:

                - theme_id
                - theme_label
                - theme_description

        mapped_themes_df : pd.DataFrame
            DataFrame mapping insights to themes.

        paper_len : int
            Target total paper length used to estimate theme summary lengths.

        Returns
        -------
        pd.DataFrame
            DataFrame containing populated theme summaries with metadata
            including:

                - thematic_summary
                - theme_id
                - theme_label
                - theme_description
                - allocated_length
                - current_length
                - perc_of_max_length

        Notes
        -----
        Themes that have no mapped insights (e.g. empty "Other" or "Conflict"
        categories) are retained with empty summaries to preserve schema
        consistency.
        """

        # Calculate the estimated lengths for each theme based on the number of insights mapped to them and merge this info back to the theme schema for use in the prompt when populating themes
        # This is only done if the columsn do not already exist, because later we will iterate on this and in those subsequent cases we just amend the allocated length manually
        # Normalise the id columsn as they come back from the LLM so could be str
        # Copy the df because this could be called on a corpus_state object
        
        schema_df = schema_df.copy()
        schema_df["question_id"] = schema_df["question_id"]
        schema_df["theme_id"] = schema_df["theme_id"].astype(int)
        
        if "allocated_length" not in schema_df.columns:
            schema_df = schema_df.merge(
            self._estimate_theme_lengths(paper_len),
            on="theme_id",
            how="left"
        )

        # Iterate over the themes from the schema to get the data for the LLM call
        populated_themes = []
        for idx, row in schema_df.iterrows():
            print(f"Populating theme {idx + 1} of {schema_df.shape[0]}...")
            rq_id = row["question_id"]
            rq_text = row["question_text"]
            theme_id = row["theme_id"]
            theme_label = row["theme_label"]
            theme_description = row["theme_description"]
            allocated_length = row["allocated_length"]
            # Get the insight ids for the specific question and theme
            insight_ids = mapped_themes_df[
                (mapped_themes_df["question_id"] == rq_id) & 
                (mapped_themes_df["theme_id"] == theme_id)
            ]["insight_id"].tolist()
            # Get the insight text from those insight ids
            insights = self.corpus_state.insights[self.corpus_state.insights["insight_id"].isin(insight_ids)]["insight"].tolist()
            # Check if insights are zero (i.e. an empty conflicts or other catergory got returned by the LLM). If so populate with an empty row
            if len(insights) == 0:
                no_insight_df = pd.DataFrame([{
                    "thematic_summary": "",
                    "question_id": rq_id,
                    "theme_id": theme_id,
                    "theme_label": theme_label,
                    "theme_description": theme_description,
                    "allocated_length": allocated_length
                }])
                populated_themes.append(no_insight_df)
                continue
            insights_str = "\n".join(insights)

            # Get the theme type to pass to the sys prompt to tailor the prompt to the theme type: general, conflict or other.
            if theme_label.strip().lower() == "conflicts":
                theme_type = "conflicts"
            elif theme_label.strip().lower() == "other":
                theme_type = "other"
            else:
                theme_type = "general"
                
            # Build the prompt
            sys_prompt = Prompts().populate_themes(theme_len=allocated_length, theme_type=theme_type)
            user_prompt = (
                f"RESEARCH QUESTION: {rq_text}\n"
                f"THEME LABEL: {theme_label}\n"
                f"THEME DESCRIPTION: {theme_description}\n"
                f"INSIGHTS TO SYNTHESIZE:\n"
                f"{insights_str}\n\n"
            )
            fall_back = {"thematic_summary": ""}

            json_schema = {
                "name": "theme_populator",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "thematic_summary": {"type": "string"}
                    },
                    "required": ["thematic_summary"],
                    "additionalProperties": False
                }
            }
            # Call the LLM
            response = utils.call_chat_completion(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                llm_client=self.llm_client,
                ai_model=self.ai_model,
                fall_back=fall_back,
                return_json=True,
                json_schema=json_schema
            )
            
            # Get the summary from the response and tag with metadata in a dataframe
            thematic_summary = pd.DataFrame([response.get("thematic_summary", "")], columns=["thematic_summary"])
            thematic_summary["question_id"] = rq_id
            thematic_summary["theme_id"] = int(theme_id)
            thematic_summary["theme_label"] = theme_label
            thematic_summary["theme_description"] = theme_description
            thematic_summary["allocated_length"] = allocated_length

            # Get the length of the summary in words and calculate the percentage of the allocated length that this summary represents
            thematic_summary["current_length"] = len(thematic_summary["thematic_summary"].iloc[0].split())
            thematic_summary["perc_of_max_length"] = thematic_summary["current_length"] / allocated_length if allocated_length > 0 else None
            
            # Append the result to the list of dfs the loop is producing, which will be concatenated at the end
            populated_themes.append(thematic_summary)
        # Concat the final list of dfs and return
        populated_themes_df = pd.concat(populated_themes, ignore_index=True)
        # Sort the values by question id and theme id so that they are in the same order as the schema and therefore able to be exported to the narrative
        populated_themes_df["theme_id"] = populated_themes_df["theme_id"].astype(int) # Before sorting defensively make sure this is int
        populated_themes_df = populated_themes_df.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)
        return(populated_themes_df)

    def _iterative_length_check_and_expand_loop(
        self, 
        populated_themes_df: pd.DataFrame, 
        max_prop: float, 
        paper_len: int
        ) -> pd.DataFrame:  

        """
        Iteratively expand theme summaries that exceed length thresholds.

        After theme summaries are generated, this method checks whether the
        model compressed any themes too aggressively relative to their
        allocated word length. If a theme exceeds the specified proportion
        of its allocation, the user is given the option to expand the theme.

        Expansion occurs by increasing the allocated word length by 20%
        and rerunning the theme population step for the affected themes.

        Parameters
        ----------
        populated_themes_df : pd.DataFrame
            DataFrame containing populated theme summaries.

        max_prop : float
            Maximum acceptable proportion of allocated length before a theme
            is flagged for expansion.

        paper_len : int
            Target total paper length used during theme population.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame of populated themes reflecting any expansions
            performed during the iterative process.

        Notes
        -----
        The loop continues until either:

            - no themes exceed the threshold, or
            - the user chooses to accept the current summaries.                                    
        """
        # Check whether any of the populated themes exceed a certain proportion of th total allocated length
        df_len_ok, df_len_flagged = self._check_length_and_flag(populated_themes_df, max_prop)
        df_len_flagged = df_len_flagged.copy() # Copy to avoid modifying the original df with the flags, as this will be used in the loop and we want to preserve the original for reference.

        #While any themes exceed the length threshold its a sign that the LLM is undertaking more compression of the granulatiry of the theme than might be optimal (to fit length requirements). 
        # Offer the user the options to re-run these themes with a 20% expansion of the word count allocated to them.
        while df_len_flagged is not None and not df_len_flagged.empty:
            expand_count = None
            while expand_count not in ["1", "2"]:
                expand_count = input(
                    f"{df_len_flagged.shape[0]} themes exceed {max_prop*100}% of their allocated length. This suggests the model is compressing these themes through aggressive abstraction and granularity is being lost. \n"
                    "Do you want to:\n"
                    "1) re-run these themes with a 20% expansion of the word count allocated to them? \n"
                    "2) keep the current populated themes and move on to the next step? \n"
                    "Enter 1 or 2:\n"
                ).lower()

            
            if expand_count == "2":
                # If the user is happy, end the loop. Populated themes is in tact and we keep it. 
                break

            else:
                # If the user wants to expand, we re run the theme populator on the flagged themes with increased allocation and update populated themes with the longer theme summaries
                # First get the schema map for the flagged themes and expand the allocated length by 20%
                df_len_flagged["allocated_length"] = (df_len_flagged["allocated_length"] * 1.2).astype(int)
                # Then get the ids and the corresponding insights
                length_check_theme_ids = df_len_flagged["theme_id"].tolist()
                rerun_mapped_df = self.summary_state.mapped_theme_list[-1][self.summary_state.mapped_theme_list[-1]["theme_id"].isin(length_check_theme_ids)].copy()
                # Rerun on the flagged themes with the expanded length and get new summaries for those themes
                rerun_theme_ids = df_len_flagged["theme_id"].tolist()
                rerun_schema_df = (
                    self.summary_state.theme_schema_list[-1][self.summary_state.theme_schema_list[-1]["theme_id"].isin(rerun_theme_ids)] # First filter the schema to just the themes we want to rerun
                    .drop(columns=["allocated_length"], errors="ignore") # Drop allocated length as we will get the updated length from df_len_flagged 
                    .copy() # Copy so that we are not modifying corpus_state
                )
                # Now get the updated allocated length via a merge
                rerun_schema_df = ( 
                    rerun_schema_df
                    .merge(df_len_flagged[["theme_id", "allocated_length"]], 
                           on="theme_id", 
                           how="left")
                )
                # Add a sanity check to make sure the allocated length came through in the merge, as this is critical for the rerun. Fail loudly if there is a problem
                if rerun_schema_df["allocated_length"].isna().any():
                    raise ValueError(
                        "Expanded allocated_length missing for some rerun themes."
                    )
                
                # Rerun the theme populater on the theme_schema_df
                rerun_populated_themes_df = self._run_theme_pop(rerun_schema_df, rerun_mapped_df, paper_len)
                rerun_len_ok, df_len_flagged = self._check_length_and_flag(rerun_populated_themes_df, max_prop)
                # If this produced any themes for which the length is now ok we replace the odl summaries in populated_theme_list with these updates
                if rerun_len_ok.shape[0] > 0:
                    # Since theme_id is globally unique i can just replace on it
                    # get the theme_ids that are now ok
                    corrected_theme_ids = rerun_len_ok["theme_id"].tolist()
                    # Drop those themes from the populated_theme_df
                    populated_themes_df = populated_themes_df[~populated_themes_df["theme_id"].isin(corrected_theme_ids)]
                    # Concat with the rerun_len_ok_themes
                    populated_themes_df = pd.concat([populated_themes_df, rerun_len_ok], ignore_index=True)
                    
        # Make sure the final df of populated themes is in the same order as the theme schema for easier comparison and so that it can be exported to the narrative in the correct order
        populated_themes_df["theme_id"] = populated_themes_df["theme_id"].astype(int) # Before sorting defensively make sure this is int
        populated_themes_df = populated_themes_df.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)
        return(populated_themes_df)
   
    def populate_themes(self, 
                        paper_len: int = 8000, 
                        max_prop: float = 0.9, 
                        force: bool = False) -> pd.DataFrame:
        
        """
        Generate narrative summaries for each theme.

        This method synthesizes insights mapped to each theme into coherent
        narrative summaries. The resulting thematic summaries form the core
        narrative structure of the synthesis output.

        Parameters
        ----------
        paper_len : int, default=8000
            Approximate total desired word length of the final synthesis
            document.

        max_prop : float, default=0.9
            Threshold proportion of allocated theme length used to detect
            overly compressed summaries.

        force : bool, default=False
            If True, bypass validation and state checks and perform a fresh
            population run. This mode is intended for development and testing.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the populated theme summaries.

        Raises
        ------
        ValueError
            If no insight-to-theme mappings exist.

        Workflow
        --------
        The method performs the following steps:

            1. Validate that insights have been mapped to themes.
            2. Optionally realign pipeline state if previous population
            passes exist.
            3. Generate theme summaries from mapped insights.
            4. Iteratively expand themes that exceed compression thresholds.
            5. Persist the resulting summaries in `SummaryState`.

        Notes
        -----
        Theme summaries are stored in `summary_state.populated_theme_list`
        to preserve the full history of synthesis passes.
        """
        
        # Check that themes have been mapped before we try to populate
        if not self.summary_state.mapped_theme_list:
            raise ValueError("No mapped themes found. Please run map_insights_to_themes() first.")
        
        #######
        # FORCE FLAG LOGIC
        #######
        # Check whether force flag is enabled. If so popylate the themes and append without running any state checks.
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and populating themes. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            # populate the themes with the insights mapped to them
            populated_themes_df = self._run_theme_pop(
                schema_df=self.summary_state.theme_schema_list[-1].copy(),
                mapped_themes_df=self.summary_state.mapped_theme_list[-1].copy(),
                paper_len=paper_len
            )
            # Iteratively expand the theme lenght until the user is satisfied
            populated_themes_df = self._iterative_length_check_and_expand_loop(
                populated_themes_df=populated_themes_df,
                max_prop=max_prop,
                paper_len=paper_len
            )

            # Append, save and return
            self.summary_state.populated_theme_list.append(populated_themes_df)
            self.summary_state.save()
            return self.summary_state.populated_theme_list[-1]
        
        #######
        # FORCE FLAG LOGIC/ENDS
        #######

        
        mapped_len = len(self.summary_state.mapped_theme_list)
        populate_len = len(self.summary_state.populated_theme_list)

        # If populated themes exist, offer choice
        if populate_len > 0:
            new = None
            while new not in ["1", "2"]:
                new = input(
                    "Populated themes already exist on disk. Do you want to:\n"
                    "(1) reload existing populated themes\n"
                    "(2) repopulate themes based on the current theme schema and mapped insights. This will realign mapping and population history.\n"
                    "Enter 1 or 2:\n"
                ).strip()
            if new == "1":
                print(
                    "Populated themes loaded. Inspect them via variable.populated_theme_list[-1]\n"
                )
                return(None) # Return to exit the function and avoid re-running the population
            
            # Realign to last coherent pre-population corpus_state
            min_len = min(mapped_len - 1, populate_len)
            self.summary_state.mapped_theme_list = self.summary_state.mapped_theme_list[:min_len + 1]
            self.summary_state.populated_theme_list = self.summary_state.populated_theme_list[:min_len]
            
        else:
            # No populated themes exist — ensure we are in valid pre-population corpus_state
            if mapped_len < 1:
                raise ValueError("No mapped themes available to populate from. Please run map_insights_to_themes() first.")
        
        
        # Populate the themes with the insights mapped to them 
        populated_themes_df = self._run_theme_pop(
            schema_df=self.summary_state.theme_schema_list[-1].copy(),
            mapped_themes_df=self.summary_state.mapped_theme_list[-1].copy(),
            paper_len=paper_len
        )
        
        # Iterate on the length of each theme and expand until the user is happy
        populated_themes_df = self._iterative_length_check_and_expand_loop(
            populated_themes_df=populated_themes_df,
            max_prop=max_prop,
            paper_len=paper_len
        )

        # Now add populated_themes_df to the populated theme list and save to disk. Return the updated themes
        self.summary_state.populated_theme_list.append(populated_themes_df)
        self.summary_state.save()
        return self.summary_state.populated_theme_list[-1]


    def _identify_orphans(
            self, 
            checked_insights_df, 
            mode, 
            batch_size
        ) -> pd.DataFrame:

        """
        Identify orphan insights not reflected in thematic summaries.

        This method audits each populated theme summary to determine whether
        all insights assigned to that theme are represented in the narrative
        synthesis. Insights that are not reflected in the summary are labeled
        as "orphans".

        The function operates in batches and supports resumable execution.
        Intermediate progress is written to a pickle file so that long-running
        audit operations can safely recover from interruptions.

        Parameters
        ----------
        checked_insights_df : pd.DataFrame or None
            DataFrame containing insights already checked during a previous
            run. Used when resuming an interrupted audit process.

        mode : str
            Determines how orphan results will be incorporated into the
            pipeline state.

            Allowed values:

                "replace" — overwrite the most recent orphan audit
                "append"  — append a new orphan audit to the history

        batch_size : int
            Number of insights checked in each LLM call.

        Returns
        -------
        pd.DataFrame
            DataFrame containing orphan insights with the following columns:

                - question_id
                - theme_id
                - insight_id
                - found (False for orphan insights)

        Notes
        -----
        A state fingerprint is stored alongside progress checkpoints to
        ensure that the corpus and summary states have not changed between
        resume attempts.
        """


        if mode not in ["replace", "append"]:
            raise ValueError("Mode must be either 'replace' or 'append'.")

        # Pickle path attribute gets set in self.address_orphans()
        self.orphan_pickle_resume_path
        # Get the state meta for the resume checks (so we don't recompute this for every batch in the loop))
        state_meta = {
            "corpus_hash": self.corpus_state.fingerprint(),
            "summary_hash": self.summary_state.fingerprint()
        }

        total_batches_to_check = math.ceil(len(self.corpus_state.insights) / batch_size)
        count = (checked_insights_df.shape[0] // batch_size) + 1 if checked_insights_df is not None else 0

        # Set the output of the loop as either the recovered df if it exists or as an empty df to populate if it does not
        checked_insights_df = pd.DataFrame(columns=["question_id", "theme_id", "insight_id", "found"]) if checked_insights_df is None else checked_insights_df
        checked_insight_id_list = checked_insights_df["insight_id"].tolist() if not checked_insights_df.empty else []
        
        # Now call the loop on these temp dataframes which will allow me to skip the insights that have already been checked and saved in the checked_insights_df which is being updated and saved to pickle as we go to allow for resumption if the process is interupted
        theme_map = self.summary_state.mapped_theme_list[-1].copy() # we need this in the loop, but to prevent copying each loop, i put it here
        for _, row in self.summary_state.populated_theme_list[-1].iterrows():
            # This runs question by question so first we iterate over those
            t_id, q_id, thematic_summary = row["theme_id"], row["question_id"], row["thematic_summary"]
            # Then we get the insights that were initially allocated to that question - as we are checking whether these got allocated
            relevant_ids = theme_map[(theme_map["theme_id"] == t_id) & (theme_map["question_id"] == q_id)]["insight_id"].tolist()
            relevant_insights = self.corpus_state.insights[
                (self.corpus_state.insights["insight_id"].isin(relevant_ids)) & # Get the insight text for the insight_ids that were mapped
                (~self.corpus_state.insights["insight_id"].isin(checked_insight_id_list)) # SKIP checked insights
            ].dropna(subset=["insight_id"]).copy()

            for i in range(0, len(relevant_insights), batch_size):
                print(f"Checking insights for theme {t_id} and question {q_id}, batch {count} of {total_batches_to_check}...")
                count += 1
                batch = relevant_insights.iloc[i : i + batch_size]
                insight_str = "\n".join(f"{rid}: {itxt}" for rid, itxt in zip(batch["insight_id"], batch["insight"]))

                sys_prompt = Prompts().identify_orphans()
                user_prompt = (
                    "THEMATIC SUMMARY:\n"
                    f"{thematic_summary}\n\n"
                    "SOURCE INSIGHTS:\n"
                    f"{insight_str}\n\n"
                )
                fall_back = {"mentioned_insight_ids": []}
                json_schema = {
                    "name": "mention_audit",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "mentioned_insight_ids": {
                            "type": "array",
                            "description": "The list of unique IDs for the insights that are reflected in the thematic summary.",
                            "items": {
                            "type": "string"
                            }
                        }
                        },
                        "required": ["mentioned_insight_ids"],
                        "additionalProperties": False
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
                # get all the found insights as sets for subtraction to get missed insights (i.e. orphans)
                insights_found = response.get("mentioned_insight_ids", [])
                insights_found_set = set(insights_found)
                insights_found_set = {str(i) for i in insights_found}
                # Get the batch to a set of strings to match the found insights set
                batch_set = set(batch["insight_id"].tolist())
                batch_set = {str(i) for i in batch["insight_id"]}
                # Get the missed insights - the orphans
                insights_missed_set = batch_set - insights_found_set
                insights_missed = list(insights_missed_set)
                # create separate dfs with diff bool conditions for found
                batch_found_df = pd.DataFrame({
                    "question_id": q_id,
                    "theme_id": t_id,
                    "insight_id": list(insights_missed),
                    "found": False
                })   
                batch_missed_df = pd.DataFrame({
                    "question_id": q_id,
                    "theme_id": t_id,
                    "insight_id": insights_found,
                    "found": True
                })
                # Get all the results together for the batch
                batch_results_df = pd.concat([batch_found_df, batch_missed_df], ignore_index=True)
                # Append the batch to the list
                checked_insights_df = pd.concat([checked_insights_df, batch_results_df], ignore_index=True)

                # Then bundle into a dict that can be pickled. Mode keeps track of whether the resume path is intended to resume or append
                checked_insights_df_meta_state_mode = {
                    "checked_insights_df": checked_insights_df,
                    "state_meta": state_meta,
                    "mode": mode
                }
                os.makedirs(config.PICKLE_SAVE_LOCATION, exist_ok=True)
                # Pickle the results of the batch so that if the process is interrupted we can resume from the last batch completed without losing all progress. 
                utils.safe_pickle(checked_insights_df_meta_state_mode, self.orphan_pickle_resume_path)

        orphans_df = checked_insights_df[checked_insights_df["found"] == False].copy()
        orphans_df["theme_id"] = orphans_df["theme_id"].astype(int) # make sure this is int for merging and comparison with the theme schema
        return(orphans_df)
    
    def _integrate_orphans(
        self, 
        orphans_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Reintegrate orphan insights into theme summaries.

        This method updates thematic summaries by incorporating insights that
        were identified as missing during the orphan audit. Each theme is
        processed independently.

        For themes containing orphan insights, the LLM receives:

            - the original theme summary
            - the theme description
            - the orphan insights

        The model is then asked to update the summary so that the missing
        insights are reflected in the narrative.

        Parameters
        ----------
        orphans_df : pd.DataFrame
            DataFrame containing insights identified as missing from theme
            summaries.

        Returns
        -------
        pd.DataFrame
            Updated theme summaries with orphan insights incorporated.

        Notes
        -----
        Themes without orphan insights are left unchanged.
        """

        updated_summary_df_lst = []
        
        # --- Added Counter Logic ---
        populated_themes = self.summary_state.populated_theme_list[-1]
        total_themes = len(populated_themes)
        count = 1
        # ---------------------------

        # Iterate over your populated themes (one row per theme)
        for _, row in populated_themes.iterrows():
            theme_id = int(row["theme_id"])
            theme_label = row["theme_label"]
            question_id = row["question_id"]
            question_text = self.corpus_state.questions[self.corpus_state.questions["question_id"] == question_id]["question_text"].iloc[0] # Get the question text from the corpus_state questions df using the question id in the row
            theme_description = row["theme_description"]
            thematic_summary = row["thematic_summary"] # Stuck to your name here
            
            # Check if this specific theme has orphans in the orphan_df
            theme_orphans = orphans_df[(orphans_df["theme_id"] == theme_id) & (orphans_df["question_id"] == question_id)]
            
            if not theme_orphans.empty:
                print(f"Integrating orphans for theme {count} of {total_themes} (Theme ID: {theme_id})...")
                
                # Fetch the actual text for the orphans from self.corpus_state.insights
                orphan_data = self.corpus_state.insights[self.corpus_state.insights["insight_id"].isin(theme_orphans["insight_id"])]
                orphan_insights_for_theme_str = "\n".join([f"- {r['insight']}" for _, r in orphan_data.iterrows()])
                
                if thematic_summary != "" and thematic_summary is not None:
                    sys_prompt = Prompts().integrate_orphans()
                    user_prompt = (
                        f"RESEARCH QUESTION: {question_text}\n"
                        f"THEME LABEL: {theme_label}\n"
                        "THEME DESCRIPTION:\n"
                        f"{theme_description}\n"
                        "ORIGINAL SUMMARY:\n"
                        f"{thematic_summary}\n"
                        "ORPHAN INSIGHTS:\n"
                        f"{orphan_insights_for_theme_str}\n\n"
                    )
                    fall_back = {"updated_summary": thematic_summary}
                    json_schema = {
                        "name": "orphan_integrator",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "updated_summary": {"type": "string"}
                            },
                            "required": ["updated_summary"],
                            "additionalProperties": False
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
                    
                    # Create the result row using your schema
                    updated_row = pd.DataFrame([{
                        "thematic_summary": response.get("updated_summary", thematic_summary),
                        "question_id": question_id,
                        "theme_id": int(theme_id),
                        "theme_label": theme_label,
                        "theme_description": theme_description,
                        "question_text": question_text
                    }])
                else:
                    updated_row = pd.DataFrame([row])
            else:
                # If no orphans for this theme, just use the original row
                print(f"No orphans found for theme {count} of {total_themes}. Skipping integration.")
                updated_row = pd.DataFrame([row])

            updated_summary_df_lst.append(updated_row)
            count += 1 # Increment counter
                
        # Final result contains one row per theme with orphans integrated
        theme_no_orphans = pd.concat(updated_summary_df_lst, ignore_index=True)
        # Sort the values by question id and theme id so that they are in the same order as the schema and therefore able to be exported to the narrative in the correct order
        theme_no_orphans["theme_id"] = theme_no_orphans["theme_id"].astype(int) # Before sorting defensively make sure this is int
        theme_no_orphans = theme_no_orphans.sort_values(by=["question_id", "theme_id"]).reset_index(drop=True)

        return(theme_no_orphans)

    def _get_orphans_and_updated_summary(
            self, 
            checked_insights_df, 
            mode,
            batch_size
        ) -> pd.DataFrame:
        """
        Run the orphan audit and integrate any missing insights.

        This method coordinates the two main steps of the orphan handling
        stage:

            1. Identify orphan insights not reflected in theme summaries.
            2. Update summaries to incorporate those insights.

        Parameters
        ----------
        checked_insights_df : pd.DataFrame or None
            Previously processed insights used during resume operations.

        mode : str
            Determines how orphan results should be stored in the summary
            state ("replace" or "append").

        batch_size : int
            Number of insights processed per audit batch.

        Returns
        -------
        tuple
            (orphans_df, updated_summary_df)

            orphans_df
                DataFrame containing orphan insights.

            updated_summary_df
                Updated theme summaries after orphan integration.

        Notes
        -----
        If no orphan insights are detected, the existing populated themes
        are returned unchanged.
        """

        orphans_df = self._identify_orphans(checked_insights_df=checked_insights_df, mode = mode, batch_size=batch_size)
        if orphans_df.shape[0] == 0:
            print("No orphans identified. All insights mapped to themes are reflected in the thematic summaries.")
            # return the last populated theme as nothing is being updated
            updated_summary_df = self.summary_state.populated_theme_list[-1]
        else:
            print(f"{orphans_df.shape[0]} orphan insights identified. Running integration process to integrate them back into the thematic summaries...")
            updated_summary_df = self._integrate_orphans(orphans_df)
        return(orphans_df, updated_summary_df)


    def address_orphans(
            self, 
            force = False,
            batch_size = 75
        ) -> pd.DataFrame:
        """
        Perform the orphan audit and integration stage of the synthesis pipeline.

        This method ensures that all insights mapped to themes are represented
        in the corresponding thematic summaries. If insights are missing,
        they are reinserted into the summaries through an LLM-assisted
        integration process.

        The method includes three operational modes:

            1. Resume mode
            If a partial orphan audit exists on disk, the user can resume
            the process from the last checkpoint.

            2. Normal mode
            The method determines whether to append a new orphan audit or
            replace the most recent one depending on pipeline state.

            3. Force mode
            All validation and resume checks are bypassed and a new orphan
            audit is appended to the state.

        Parameters
        ----------
        force : bool, default=False
            If True, bypass pipeline sequencing safeguards and execute the
            orphan process regardless of current state.

        batch_size : int, default=75
            Number of insights evaluated in each orphan audit batch.

        Returns
        -------
        pd.DataFrame
            DataFrame containing orphan insights identified during the audit.

        Raises
        ------
        ValueError
            If populated themes do not exist or if a resume attempt detects
            a state mismatch.

        Workflow
        --------
        The method performs the following steps:

            1. Validate that theme summaries exist.
            2. Check for resumable orphan audit progress.
            3. Identify orphan insights.
            4. Integrate orphan insights into summaries.
            5. Update the summary state and persist results.

        Notes
        -----
        The orphan detection stage is designed to reduce silent omission of
        insights during synthesis. If orphan insights are identified, the
        summaries are regenerated with those insights incorporated before
        the next iteration of theme schema generation.
        """

        # The logic for this function is as follows:
        # First check if there is a resume file, if so resume according to the mode passed originally
        # Then run a force flag, which passes the mode append and skips the sequencing validation checks
        # Finally, if no force and no reusme, check the state to determine the sequencing needs and offer the user to replace or extend, and then pass that mode to the call - so that it can resume correctly if it crashes

        # Make sure the required state elements exist to check for orphans
        if len(self.summary_state.populated_theme_list) == 0:
            raise ValueError("No populated themes found. Please run populate_themes() first.")

        # ####
        # RESUME LOGIC
        # ### 
        # First check resume logic - this effectively sets the mode and the checked_insights_df
        checked_insights_df = None
        self.orphan_pickle_resume_path = os.path.join(config.PICKLE_SAVE_LOCATION, "orphan_check_in_progress.pickle")
        if os.path.exists(self.orphan_pickle_resume_path):
            resume = None
            while resume not in ["1", "2"]:
                resume = input("A partial orphan identification process was detected. Do you want to:\n"
                               "1) resume from the last saved point? \n"
                               "2) start a new orphan identification process? \n"
                               "Enter 1 or 2:\n").lower()
            if resume == "1":
                with open(self.orphan_pickle_resume_path, "rb") as f:
                    checked_insights_df_meta_state = pickle.load(f)
                    # Check that the pickled state matches the current state
                    pickled_state = checked_insights_df_meta_state["state_meta"]
                    current_state = {
                        "corpus_hash": self.corpus_state.fingerprint(),
                        "summary_hash": self.summary_state.fingerprint()
                    }
                    if pickled_state != current_state:
                        raise ValueError(
                            "The corpus state or summary state has changed between resume. Resume invalid"
                            "Either choose a fresh start or if you know what you are doing use force = True" 
                            )
                    # Load the checked insights data frame from the pickle to resume the process
                    checked_insights_df = checked_insights_df_meta_state["checked_insights_df"]
                    mode = checked_insights_df_meta_state["mode"]
                    print("Resuming orphan identification process from last saved point...")
                    orphans_df, updated_summary_df = self._get_orphans_and_updated_summary(checked_insights_df=checked_insights_df, mode=mode, batch_size=batch_size)
                    self.summary_state.populated_theme_list[-1] = updated_summary_df
                    if mode == "replace":
                        self.summary_state.orphan_list[-1] = orphans_df
                    elif mode == "append":
                        self.summary_state.orphan_list.append(orphans_df)
                    
                    # Clean up the resume pickle as the process is complete and we want to void a resume trigger if we run again
                    os.remove(self.orphan_pickle_resume_path)
                    self.summary_state.save()
                    return(self.summary_state.orphan_list[-1])
 
            else:
                print("Starting new orphan identification process and deleting the in progress pickle...")
                os.remove(self.orphan_pickle_resume_path)


        # #####
        # FORCE FLAG LOGIC
        # #####
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and addressing orphans. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            mode = "append" # This is a bit meaningless because with force we are always appending, but this makes it clear that we are not replacing any existing orphan audits and just appending a new one on top of the existing state regardless of the current state of the orphan list.       
            orphans_df, updated_summary_df = self._get_orphans_and_updated_summary(checked_insights_df=None, mode="append", batch_size=batch_size) # Mode is append because with force we are always appending. We set it here in case the system crashes, in which case the recover will propagate the append mode. We actually don't need to access the mode before this return as we just append in force
            # We always repair the populated themes with the integrated version. No change under force
            self.summary_state.populated_theme_list[-1] = updated_summary_df
            # We append orphans because this is force and we always append
            self.summary_state.orphan_list.append(orphans_df)

            # Clean up           
            if os.path.exists(self.orphan_pickle_resume_path):
                os.remove(self.orphan_pickle_resume_path)
            self.summary_state.save()
            return(self.summary_state.orphan_list[-1])
        
        # #####
        # SEQUENCING LOGIC
        # #####

        # --- Realign orphan list to coherent corpus_state ---
        min_len = min(len(self.summary_state.populated_theme_list), len(self.summary_state.orphan_list))
        self.summary_state.orphan_list = self.summary_state.orphan_list[:min_len]

        # --- Determine behavior ---
        # Essentially if the lenght of orphan and theme lists are the same the orphan has already been created for the latest theme list. If the user want to run again, we need to replace
        # the last orphan rather than extending it. If orphans is behind, then we just extend the list
        if len(self.summary_state.populated_theme_list) == len(self.summary_state.orphan_list):
            # Orphan already exists for latest populate
            choice = None
            while choice not in ["1", "2"]:
                choice = input(
                    "Orphan handling already exists for the latest populated themes.\n"
                    "(1) Reload existing orphan audit\n"
                    "(2) Re-run orphan identification and incorporation\n"
                    "Enter 1 or 2:\n"
                ).strip()

            if choice == "1":
                print("Latest orphan audit loaded.")
                return None
            # If we are not reloading, and populated themes and orphans are aligned, then we replace the most recent orphan audit with a new one - we are essentially updating the audit, so mode is replace
            mode = "replace" 
            # Otherwise if orphans is behind populated themes we are adding the new audit to align the state
        else:
            mode = "append"

        # Get the orphans and the updated summaries based on whether we are replacing or appending
        orphans_df, updated_summary_df = self._get_orphans_and_updated_summary(checked_insights_df=checked_insights_df, mode=mode, batch_size=batch_size) #Checked insights is None, based on false eval for resume. Mode propagates in case there is a crash
        self.summary_state.populated_theme_list[-1] = updated_summary_df
        if mode == "replace":
            self.summary_state.orphan_list[-1] = orphans_df
        elif mode == "append":
            self.summary_state.orphan_list.append(orphans_df)

        # Now clean up and save
        # Delete the resume pickle as the process is complete and we want to void a resume trigger if we run again
        if os.path.exists(self.orphan_pickle_resume_path):
            os.remove(self.orphan_pickle_resume_path)
        self.summary_state.save()
        return(self.summary_state.orphan_list[-1])

    def _llm_redundancy_check(self):
        """
        Reduce redundancy across theme summaries.

        This method performs a sequential redundancy pass over the populated
        theme summaries. Each theme is processed in order and compared against
        the summaries of previously processed themes within the same research
        question.

        The LLM receives:

            - the research question
            - the already-refined summaries of preceding themes
            - the current theme text

        It then rewrites the current theme summary so that information already
        covered by earlier themes is minimized while preserving all unique
        content relevant to the theme.

        The process is sequential within each research question, meaning that
        each refined theme becomes part of the context for refining subsequent
        themes.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the refined theme summaries with the same
            schema as the populated theme summaries.

        Notes
        -----
        The redundancy pass is applied after the thematic synthesis process
        has stabilized. It does not modify the underlying insight-to-theme
        mappings and is intended only to improve readability and reduce
        repeated information across themes.
        """

        ordered_themes = self.summary_state.populated_theme_list[-1].sort_values(by=["question_id", "theme_id"]).reset_index(drop=True).copy()
        refined_rows = []

        total_themes = ordered_themes.shape[0]
        count = 1

        for q_id, q_group in ordered_themes.groupby("question_id"):
            previous_theme_text = "" # Reset for each Research Question
            
            for _, row in q_group.iterrows():
                print(f"Addressing redundancy for theme {count} of {total_themes}")
                # Prepare the Payload
                question_text = row["question_text"]
                theme_label = row["theme_label"]
                current_theme_text = row["thematic_summary"]

                # System Prompt (Your structural version)
                sys_prompt = Prompts().address_redundancy()
                
                # User Prompt (Matching your INPUT FORMAT)
                user_prompt = (
                    f"RESEARCH QUESTION: {question_text}\n"
                    f"PREVIOUSLY CLEANED THEMES:\n{previous_theme_text if previous_theme_text else 'None.'}\n\n"
                    f"CURRENT THEME LABEL: {theme_label}\n"
                    f"CURRENT THEME TEXT TO REFINE:\n{current_theme_text}"
                )
                
                fall_back = {"refined_theme": ""}

                # Execute the reduction
                response = utils.call_chat_completion(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    llm_client=self.llm_client,
                    ai_model=self.ai_model,
                    return_json=True,
                    fall_back=fall_back,
                    json_schema={
                        "name": "redundancy_reduction",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"refined_theme": {"type": "string"}},
                            "required": ["refined_theme"],
                            "additionalProperties": False
                        }
                    }
                )

                refined_theme = response["refined_theme"]
                
                # Update context for the next theme in this RQ
                previous_theme_text += f"\n\n{refined_theme}"

                # Preserve the row data
                refined_row = row.copy()
                refined_row["thematic_summary"] = refined_theme
                refined_rows.append(refined_row)
                count += 1

        # Push to State
        refined_df = pd.DataFrame(refined_rows)
        refined_df["theme_id"] = refined_df["theme_id"].astype(int)
        return(refined_df)

    def address_redundancy(self, 
                           force = False) -> pd.DataFrame:
        """
        Perform the final redundancy reduction pass on theme summaries.

        This method runs a sequential redundancy check across the most recent
        populated themes to reduce repeated information between theme summaries.

        Parameters
        ----------
        force : bool, default=False
            If True, bypass sequencing validation and run the redundancy
            reduction regardless of pipeline state.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the refined theme summaries after redundancy
            reduction.

        Raises
        ------
        ValueError
            If populated themes do not exist or if orphan handling has not
            been completed prior to the redundancy pass.

        Workflow
        --------
        The method proceeds through the following steps:

            1. Validate that populated themes exist.
            2. Ensure orphan handling has been completed.
            3. Optionally reload an existing redundancy pass.
            4. Run redundancy reduction using the LLM.
            5. Store the refined summaries in `SummaryState.redundancy_list`.

        Notes
        -----
        The redundancy pass represents the final transformation applied to
        theme summaries before rendering. It improves narrative clarity by
        removing repeated material while preserving all distinct information
        contained in the themes.
        """
        # Make sure this can run
        if not self.summary_state.populated_theme_list:
            raise ValueError("No populated themes available for redundancy pass.")
        # Force logic: run regardless and populate redundancy attribute and return
        if force:
            print("WARNING: Force flag is True: Skipping validation and resumption checks and addressing redundancy. This may cause the state to become unstable. This mode should be used for testing purposes only.")
            # We just run the redundancy reduction on the latest populated themes and append to the redundancy list without any checks
            refined_df = self._llm_redundancy_check()
            self.summary_state.redundancy_list = [refined_df]
            self.summary_state.save()
            return self.summary_state.redundancy_list[-1]
        
        # If not force then make sure the orphans are not smaller than populated themes - i.e. you have to have run orphans before you can run redundancy. But orphans can be larger - if you for some reason iterate orphans and redundancy, orphans will grow while everythig else stays as is
        if len(self.summary_state.orphan_list) < len(self.summary_state.populated_theme_list):
            raise ValueError("The number of orphan audits and populated themes are not aligned. Please run address_orphans() to align the state before running redundancy reduction.")

        # Now check whether we want to reload or rebuild redundancy
        if len(self.summary_state.redundancy_list) == 1:
            reload = None
            while reload not in ["1", "2"]:
                reload = input(
                    "Redundancy reduction has already been run on the latest populated themes.\n"
                    "(1) Reload existing redundancy reduction\n"
                    "(2) Re-run redundancy reduction (this will overwrite the previous redundancy pass)\n"
                    "Enter 1 or 2:\n"
                ).strip()
            if reload == "1":
                print("Latest redundancy reduction loaded.")
                return self.summary_state.redundancy_list[0] # Return to exit the function and avoid re-running the reduction    
            else:
                print("Re-running redundancy reduction on the latest populated themes...")

        # Do the redundancy check, update redundancy_list, save and return
        refined_df = self._llm_redundancy_check()
        self.summary_state.redundancy_list = [refined_df]
        self.summary_state.save()
        return self.summary_state.redundancy_list[-1]















                
                        



        


            



        # # Utility function to estimate the length of each theme based on the number of insights mapped to it, relative to the total number of insights for that research question, and allocate a proportion of the total paper length to each theme accordingly. This will be used to prompt the LLM on how much to write for each theme when we get to the population stage.
        
        
        # # Calculate the estimated lengths for each theme based on the number of insights mapped to them and merge this info back to the theme schema for use in the prompt when populating themes
        # # Prior to doing the caluclation remove any wordcounts that might have been calculated on prior runs
        # self.summary_state.theme_schema_list[-1] = self.summary_state.theme_schema_list[-1].drop(columns=["allocated_length"], errors="ignore")
        # # Then populate the new lengths and merge to the theme schema
        # self.summary_state.theme_schema_list[-1] = (
        #     self.summary_state.theme_schema_list[-1].
        #     merge(
        #         self._estimate_theme_lengths(),
        #         on=["question_id", "theme_id"],
        #         how="left"
        #         )
        #     )
        
        # populated_themes = []    
        # for _, row in self.summary_state.theme_schema_list[-1].iterrows():
        #     rq_id = row["question_id"]
        #     rq_text = row["question_text"]
        #     theme_id = row["theme_id"]
        #     theme_label = row["theme_label"]
        #     theme_description = row["theme_description"]
        #     allocated_length = row["allocated_length"]
        #     insight_ids = self.summary_state.mapped_theme_list[-1][
        #         (self.summary_state.mapped_theme_list[-1]["question_id"] == rq_id) & 
        #         (self.summary_state.mapped_theme_list[-1]["theme_id"] == theme_id)
        #     ]["insight_id"].tolist()
        #     insights = self.corpus_state.insights[self.corpus_state.insights["insight_id"].isin(insight_ids)]["insight"].tolist()
        #     if len(insights) == 0:
        #         no_insight_df = pd.DataFrame([{
        #             "thematic_summary": "",
        #             "question_id": rq_id,
        #             "theme_id": theme_id,
        #             "theme_label": theme_label,
        #             "theme_description": theme_description,
        #             "allocated_length": allocated_length
        #         }])
        #         populated_themes.append(no_insight_df)
        #         continue
        #     insights_str = "\n".join(insights)

        #     sys_prompt = Prompts().populate_themes(theme_len=allocated_length)
        #     user_prompt = (
        #         f"RESEARCH QUESTION: {rq_text}\n"
        #         f"THEME LABEL: {theme_label}\n"
        #         f"THEME DESCRIPTION: {theme_description}\n"
        #         f"INSIGHTS TO SYNTHESIZE:\n"
        #         f"{insights_str}\n\n"
        #     )
        #     fall_back = {"thematic_summary": ""}

        #     json_schema = {
        #         "name": "theme_populator",
        #         "strict": True,
        #         "schema": {
        #             "type": "object",
        #             "properties": {
        #                 "thematic_summary": {"type": "string"}
        #             },
        #             "required": ["thematic_summary"],
        #             "additionalProperties": False
        #         }
        #     }

        #     response = utils.call_chat_completion(
        #         sys_prompt=sys_prompt,
        #         user_prompt=user_prompt,
        #         llm_client=self.llm_client,
        #         ai_model=self.ai_model,
        #         fall_back=fall_back,
        #         return_json=True,
        #         json_schema=json_schema
        #     )
            
        #     thematic_summary = pd.DataFrame([response.get("thematic_summary", "")], columns=["thematic_summary"])
        #     thematic_summary["question_id"] = rq_id
        #     thematic_summary["theme_id"] = theme_id
        #     thematic_summary["theme_label"] = theme_label
        #     thematic_summary["theme_description"] = theme_description
        #     thematic_summary["allocated_length"] = allocated_length

        #     thematic_summary["current_length"] = len(thematic_summary["thematic_summary"].iloc[0].split())
        #     thematic_summary["perc_of_max_length"] = thematic_summary["current_length"] / allocated_length if allocated_length > 0 else None
            
        #     populated_themes.append(thematic_summary)

        
        # populated_themes_df = pd.concat(populated_themes, ignore_index=True)

    


        
        
        


        # self.summary_state.populated_theme_list.append(populated_themes_df)
        # os.makedirs(self.summary_save_location, exist_ok=True)
        # for idx, df in enumerate(self.summary_state.populated_theme_list):
        #     df.to_parquet(os.path.join(self.summary_save_location, f"populated_themes_{idx+1}.parquet"), index=False)
        
        # return self.summary_state.populated_theme_list[-1]










        
        
        # # utils
        # def build_frozen_block(frozen_content: list[dict]) -> str:
        #     if not frozen_content:
        #         return "(none)\n"
        #     parts = []
        #     for t in frozen_content:
        #         parts.append(
        #             f"Theme ID: {t.get('theme_id','')}\n"
        #             f"Label: {t.get('label','')}\n"
        #             f"Criteria: {t.get('criteria','')}\n"
        #             f"Content:\n{t.get('contents','')}\n"
        #             "--- END THEME ---"
        #         )
        #     return "\n".join(parts) + "\n"

        # def build_remaining_themes_block(rq_df: pd.DataFrame, current_theme_id: str, processed_ids: set[str]) -> str:
        #     # remaining = all themes for this RQ not yet processed, excluding the current one
        #     rem = rq_df.loc[~rq_df["id"].isin(processed_ids | {current_theme_id}), ["label", "criteria"]]
        #     if rem.empty:
        #         return "(none)\n"
        #     parts = []
        #     for _, r in rem.iterrows():
        #         parts.append(
        #             f"Theme label: {r['label']}\n"
        #             f"Criteria: {r['criteria']}\n"
        #             "--- END THEME ---"
        #         )
        #     return "\n".join(parts) + "\n"

        # # guard
        # if not hasattr(self, "summary_themes"):
        #     raise ValueError("No summary themes found. Please run identify_themes() first.")

        # save_dir = self.summary_save_location
        # save_path = os.path.join(save_dir, save_file_name)

        # if os.path.exists(save_path):
        #     recover = None
        #     while recover not in ['r', 'n']:
        #         recover = input("Populated themes already exist on disk. Recover (r) or generate new (n)? ").lower()
        #     if recover == 'r':
        #         self.populated_themes = pd.read_parquet(save_path)
        #         return self.populated_themes
        #     else:
        #         print("Re-running population of themes...")

        # out_rows = []
        # total_themes = len(self.summary_themes)
        # counter = 0

        # # iterate per research question
        # for question_id, rq_df in self.summary_themes.groupby("question_id", sort=False):
        #     # reset frozen content per question to avoid leakage
        #     frozen_content: list[dict] = []
        #     processed_ids: set[str] = set()

        #     # source text for this RQ
        #     summary_text_list = self.summaries.loc[self.summaries["question_id"] == question_id, "summary"].tolist()
        #     summary_text = "\n\n".join(summary_text_list)

        #     # iterate themes for this question in the given order
        #     for _, row in rq_df.iterrows():
        #         counter += 1
        #         print(f"Populating theme {counter} of {total_themes}")

        #         question_text = row["question_text"]
        #         theme_id = row["id"]
        #         theme_label = row["label"]
        #         theme_criteria = row["criteria"]

        #         frozen_block = build_frozen_block(frozen_content)
        #         remaining_theme_block = build_remaining_themes_block(rq_df, theme_id, processed_ids)

        #         sys_prompt = Prompts().populate_themes()
        #         user_prompt = (
        #             f"Research question id: {question_id}\n"
        #             f"Research question text: {question_text}\n"
        #             "FROZEN CONTENT (read-only; text already assigned to themes):\n"
        #             f"{frozen_block}"
        #             "---CURRENT THEME TO POPULATE:---\n"
        #             f"Theme ID: {theme_id}\n"
        #             f"Theme label: {theme_label}\n"
        #             f"Criteria: {theme_criteria}\n\n"
        #             "CLUSTER SUMMARY TEXT (source material):\n"
        #             f"{summary_text}\n\n"
        #             "--- THEMES STILL TO PROCESS (context only):---\n"
        #             f"{remaining_theme_block}"
        #         )

        #         fall_back = {"question_id": question_id, "theme_id": theme_id, "assigned_content": ""}

        #         resp = utils.call_chat_completion(
        #             sys_prompt=sys_prompt,
        #             user_prompt=user_prompt,
        #             llm_client=self.llm_client,
        #             ai_model=self.ai_model,
        #             return_json=True,
        #             fall_back=fall_back,
        #         )

        #         assigned = (resp.get("assigned_content") or "").strip()

        #         out_row = {
        #             "question_id": question_id,
        #             "question_text": question_text,
        #             "theme_id": theme_id,
        #             "label": theme_label,
        #             "criteria": theme_criteria,
        #             "contents": assigned,
        #         }
        #         out_rows.append(out_row)

        #         # update frozen and processed sets
        #         frozen_content.append(out_row)
        #         processed_ids.add(theme_id)

        # output = pd.DataFrame(
        #     out_rows, columns=["question_id", "question_text", "theme_id", "label", "criteria", "contents"]
        # )

        # self.populated_themes = output
        # os.makedirs(save_dir, exist_ok=True)
        # self.populated_themes.to_parquet(save_path)
        # return self.populated_themes
        