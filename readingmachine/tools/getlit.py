"""
Corpus discovery tools for ReadingMachine.

This module provides utilities for identifying and assembling document
corpora prior to running the ReadingMachine analysis pipeline. The
classes defined here assist with literature discovery, deduplication,
coverage checking, and document organization.

Unlike the main ReadingMachine pipeline, which focuses on structured
reading and synthesis of an existing corpus, the tools in this module
are designed to help users *find* and *prepare* candidate documents.

Typical workflow
----------------

A common workflow using this module may look like:

    research questions
        ↓
    ScholarSearchString
        ↓
    AcademicLit / GreyLiterature
        ↓
    Literature (deduplication)
        ↓
    AiLiteratureCheck (coverage check)
        ↓
    DownloadManager
        ↓
    document corpus ready for ingestion

The resulting corpus can then be processed by the main ReadingMachine
pipeline:

    documents
        ↓
    ingestion
        ↓
    insight extraction
        ↓
    clustering
        ↓
    thematic synthesis

Classes
-------

ScholarSearchString
    Generate structured literature search queries from research
    questions using an LLM.

AcademicLit
    Retrieve academic publications using scholarly APIs such as
    Crossref and OpenAlex.

GreyLiterature
    Identify relevant grey literature using LLM-assisted web search.

Literature
    Deduplicate and organize retrieved literature records using exact
    and fuzzy matching techniques.

AiLiteratureCheck
    Use an LLM to identify potentially missing papers for each research
    question based on the current literature corpus.

DownloadManager
    Create a filesystem structure for managing manual document
    downloads while preserving traceability between literature records
    and document files.

Design principles
-----------------

The discovery tools are intentionally separated from the core
ReadingMachine pipeline so that the system can be used with any
pre-existing document corpus.

This separation ensures that ReadingMachine is fundamentally a
**corpus reading and synthesis framework**, rather than a literature
search tool.
"""

# Import custom libraries and modules
from readingmachine.state import CorpusState
from readingmachine.prompts import Prompts
from readingmachine import config
from readingmachine import utils

# Import standard libraries
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import pandas as pd
import json
from copy import deepcopy
from habanero import Crossref
import time 
from requests.exceptions import HTTPError
import random
import requests
import pickle
import numpy as np
import re



class ScholarSearchString:
    """
    Generate literature search strings from research questions.

    This class provides an optional corpus discovery stage that converts
    research questions into structured search strings suitable for
    academic search engines (e.g., Semantic Scholar).

    The resulting search strings are stored in `CorpusState` so that the
    document retrieval process can be integrated into the ReadingMachine
    pipeline.

    Workflow
    --------
    1. Initialize with research questions and an LLM client.
    2. Construct an initial `CorpusState` containing question identifiers.
    3. Generate structured prompts for each question.
    4. Use the LLM to produce search strings.
    5. Store generated search strings in the corpus state.

    Notes
    -----
    This component is part of the `getlit` toolset rather than the core
    ReadingMachine pipeline. It supports literature discovery but is not
    required when analyzing a pre-existing document corpus.

    """

    def __init__(
        self,
        questions: List[str],
        llm_client: Any,
        num_prompts: int = 10,
        search_engine: str = "Semantic Scholar",
        llm_model: str = "gpt-4.1",
        corpus_state: Optional[CorpusState] = None,
        messages: Optional[List[List[Dict[str, str]]]] = None,
    ) -> None:
        """
         Initialize the ScholarSearchString generator.

        Parameters
        ----------
        questions : List[str]
            List of research questions for which search queries will be
            generated.

        llm_client : Any
            LLM client instance used to send prompt requests.

        num_prompts : int, default=10
            Number of search queries to generate per research question.

        search_engine : str, default="Semantic Scholar"
            Search engine context used to guide query generation.

        llm_model : str, default="gpt-4.1"
            Name of the LLM model used for generating search strings.

        corpus_state : Optional[CorpusState]
            Existing `CorpusState` object to populate. If not provided,
            a new state object will be created.

        messages : Optional[List[List[Dict[str, str]]]]
            Pre-generated message objects. Typically left as `None` and
            generated internally.
        """
        self.questions: List[str] = questions
        self.llm_client: Any = llm_client
        self.num_prompts: int = num_prompts
        self.search_engine: str = search_engine
        self.llm_model: str = llm_model

        # Create or reuse a CorpusState object
        if corpus_state:
            self.corpus_state = corpus_state
        else:
            self.corpus_state = CorpusState(questions = self._make_state(), 
                                            insights = self._make_state())
                                            

        # Messages are generated later by `message_maker`
        self.messages: Optional[List[List[Dict[str, str]]]] = messages

    def _make_state(self) -> pd.DataFrame:
        """
        Construct the initial question state DataFrame.

        This method creates a minimal DataFrame representing the research
        questions used to generate search strings.

        Each question is assigned a unique identifier that is used
        throughout the search-string generation process.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:

            - question_id
            - question_text
        """
        corpus_state = pd.DataFrame()
        question_ids: List[str] = [f"question_{i}" for i in range(len(self.questions))]
        corpus_state["question_id"] = question_ids
        corpus_state["question_text"] = self.questions
        return corpus_state

    def searchstring_maker(self) -> List[str]:
        """
        Generate literature search queries for each research question.

        This method uses the LLM to create a set of candidate search
        strings for each research question. The generated queries are
        intended for use with academic search engines.

        For each question:

            1. A prompt is constructed containing the question text.
            2. The LLM generates multiple search queries.
            3. The queries are stored in a tidy DataFrame structure.
            4. The resulting DataFrame is written to `CorpusState.insights`.

        Returns
        -------
        List[str]
            List of generated search strings across all research questions.

        Notes
        -----
        The resulting DataFrame stored in `CorpusState.insights` contains:

            - question_id
            - question_text
            - search_string
            - search_string_id

        Each search string receives a unique identifier to maintain
        traceability within the pipeline.

        The updated `CorpusState` is saved to disk so that subsequent
        stages of the document identification workflow can resume safely.
        """
        sys_prompt: str = Prompts().question_make_sys_prompt(
            search_engine=self.search_engine,
            num_prompts=self.num_prompts,
        )
        
        # List of prompts to get back from the LLM to be populated by the for loop
        output_search_prompts = []

        # Build user prompts from question IDs and texts
        for idx, row in self.corpus_state.questions.iterrows():
            print(f"Generating search prompts for question {idx + 1} of {self.corpus_state.questions.shape[0]}")
            question_id = row["question_id"]
            question_text = row["question_text"]
            user_prompt: str = f"**QUESTION**\n{question_id}: {question_text}"
            # Generate json schema - this is small so i can do it in the loop
            json_schema = {
                    "name": "search_generation",
                    "strict": True,
                    "schema": {
                    "type": "object",
                    "properties": {
                        "search_prompts": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of search queries generated based on the user input."
                        }
                    },
                    "required": ["search_prompts"],
                    "additionalProperties": False
                    }
                }
            
            # Fall back for failed response
            fall_back = {"search_prompts": []}
            # Call llm 
            response = utils.call_chat_completion(
                llm_client=self.llm_client,
                ai_model=self.llm_model,
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                fall_back = fall_back,
                return_json=True,
                json_schema=json_schema
                )

            # procees respnse and append to output list
            prompts = response["search_prompts"]
            output_search_prompts.append(prompts)
        
        # Convert to df so that we can explode on search terms which are a list - to get insights as a tidy df
        output_search_prompts_df = pd.DataFrame({
            "question_id": self.corpus_state.questions["question_id"],
            "question_text": self.corpus_state.questions["question_text"],
            "search_string": output_search_prompts
            })

        # Explode
        output_search_prompts_df = output_search_prompts_df.explode("search_string").reset_index(drop=True)
        # Add search string id
        output_search_prompts_df["search_string_id"] = [f"search_string_{i}" for i in range(output_search_prompts_df.shape[0])]

        # update the insights in corpus_state
        self.corpus_state.insights = output_search_prompts_df
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "01_search_strings"))
        
        # Return the search strings for inspection
        return self.corpus_state.insights["search_string"].to_list()


class AcademicLit:
    """
    Retrieve academic literature using search strings stored in CorpusState.

    This class queries external scholarly APIs to identify candidate
    publications relevant to research questions. It currently supports
    multiple search providers, including:

        - Crossref
        - OpenAlex

    Search results are merged into `CorpusState.insights` so they can be
    used downstream in the ReadingMachine pipeline.

    Workflow
    --------
    1. Initialize with an existing `CorpusState` or a DataFrame of
    search strings.
    2. Query academic APIs using the stored search strings.
    3. Normalize returned metadata (title, authors, year, DOI).
    4. Merge results into the existing corpus state.
    5. Deduplicate records across search engines.

    Notes
    -----
    The purpose of this class is **corpus discovery**, not analysis.
    Retrieved publications are stored in `CorpusState` so that users
    can inspect them, filter them, and later ingest full-text documents
    into the ReadingMachine analytical pipeline.
    """

    OPENALEX_BASE = "https://api.openalex.org/works"
    OPENALEX_TIMEOUT = 30

    def __init__(self, 
                 corpus_state: Optional["CorpusState"] = None, 
                 search_strings: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the AcademicLit search interface.

        The class requires either an existing `CorpusState` object
        containing search strings or a DataFrame of search strings that
        will be converted into a valid corpus state.

        Parameters
        ----------
        corpus_state : Optional[CorpusState]
            Existing pipeline state containing the search strings used for
            literature retrieval.

        search_strings : Optional[pd.DataFrame]
            DataFrame containing search strings and associated question
            metadata.

        Notes
        -----
        The constructor validates the input format and ensures that the
        resulting state contains the required columns:

            - question_id
            - question_text
            - search_string
            - search_string_id
        """
        # Deepcopy ensures this class has its own copy of corpus_state
        self.corpus_state = deepcopy(
            utils.validate_format(
                corpus_state=corpus_state,
                injected_value=search_strings,
                state_required_cols=["question_id", "question_text", "search_string", "search_string_id"],
                injected_required_cols=["question_id", "question_text", "search_string", "search_string_id"]
            )
        )
    
    # CLASS UTILS ------------
    def _merge_search_results_with_state(self, search_results: pd.DataFrame) -> pd.DataFrame:
        """
        Merge newly retrieved literature records into the corpus state.

        This helper method combines search results from external academic
        APIs with the existing `CorpusState.insights` DataFrame.

        The method:

            1. Adds question metadata associated with each search string.
            2. Appends results to the existing corpus state.
            3. Deduplicates results across search engines.

        Parameters
        ----------
        search_results : pd.DataFrame
            DataFrame containing results returned by an academic API.

        Returns
        -------
        pd.DataFrame
            Updated insights DataFrame containing both existing and newly
            retrieved literature records.

        Notes
        -----
        Deduplication is performed using the following fields:

            - paper_title
            - paper_date
            - search_engine
            - search_string
        """
        if "search_engine" in getattr(self.corpus_state, "insights", pd.DataFrame()).columns:
            # Merge to bring in question info and concatenate with existing results
            enriched = search_results.merge(
                self.corpus_state.insights[["question_id", "question_text", "search_string_id", "search_string"]],
                on="search_string",
                how="left"
            )
            merged = pd.concat([self.corpus_state.insights, enriched], ignore_index=True)
        else:
            # First engine call — merge just to add question info
            merged = search_results.merge(
                self.corpus_state.insights,
                on="search_string",
                how="left"
            )

        merged = merged.drop_duplicates(
            subset=["paper_title", "paper_date", "search_engine", "search_string"]
        ).reset_index(drop=True)

        return merged
    
        # END CLASS UTILS ---------------

    def search_crossref(self, num_results: int = 10) -> pd.DataFrame:
        """
        Search Crossref for publications matching the stored search strings.

        Each search string is submitted to the Crossref API, and the
        returned publication metadata is extracted and normalized before
        being merged into the corpus state.

        Parameters
        ----------
        num_results : int, default=10
            Maximum number of results returned per search string.

        Returns
        -------
        pd.DataFrame
            Updated `CorpusState.insights` DataFrame containing retrieved
            Crossref records.

        Retrieved Fields
        ----------------
        For each publication the following metadata is extracted:

            - paper_title
            - paper_author
            - paper_date
            - doi
            - search_string
            - search_engine
            - paper_id

        Notes
        -----
        Crossref requests are retried once if the API returns a rate-limit
        or service-unavailable response (HTTP 429 or 503).
        """
        # Must be a full email address
        mailto = os.getenv("EMAIL_DOMAIN")
        cr = Crossref(mailto=mailto)

        output: Dict[str, List] = {
            "search_string": [],
            "paper_title": [],
            "paper_author": [],
            "paper_date": [],
            "doi": []
        }

        for search_string in self.corpus_state.insights["search_string"]:
            # one request; optionally retry once on 503/429
            for attempt in (1, 2):
                try:
                    res = cr.works(query=search_string, limit=num_results)
                    break
                except HTTPError as e:
                    code = getattr(e.response, "status_code", None)
                    if code in (503, 429) and attempt == 1:
                        delay = 5 + random.random() * 5
                        print(f"Crossref {code}; retrying in {delay:.1f}s…")
                        time.sleep(delay)
                        continue
                    raise

            items = res.get("message", {}).get("items", []) if res else []

            for item in items:
                # Title (list -> first string)
                title = (item.get("title") or ["No title found"])[0]
                output["paper_title"].append(title)

                # Authors (safe formatting)
                authors = []
                for a in item.get("author", []) or []:
                    fam = a.get("family", "")
                    giv = a.get("given", "")
                    if fam or giv:
                        initials = (giv[:1] + ".") if giv else ""
                        authors.append(f"{fam}, {initials}".strip(", "))
                output["paper_author"].append(authors or None)

                # Year (print -> online -> issued -> NA)
                def _year(parts):
                    try:
                        return parts[0][0]
                    except Exception:
                        return None
                year = None
                if "published-print" in item:
                    year = _year(item["published-print"].get("date-parts", []))
                if year is None and "published-online" in item:
                    year = _year(item["published-online"].get("date-parts", []))
                if year is None and "issued" in item:
                    year = _year(item["issued"].get("date-parts", []))
                output["paper_date"].append(year if year is not None else pd.NA)

                # DOI
                output["doi"].append(item.get("DOI", pd.NA))

                output["search_string"].append(search_string)

        output_df = pd.DataFrame(output)
        output_df["search_engine"] = "Crossref"
        output_df["paper_id"] = [f"crossref_paper_{i+1}" for i in range(len(output_df))]
        output_df["paper_author"] = [";".join(authors) if isinstance(authors, list) else authors for authors in output["paper_author"]]

        # Merge results into corpus_state
        self.corpus_state.insights = self._merge_search_results_with_state(output_df)
        return self.corpus_state.insights
    
    @staticmethod
    def _openalex_authors(authorships) -> List[str]:
        """
        Extract author names from OpenAlex authorship records.

        Parameters
        ----------
        authorships : list
            List of authorship dictionaries returned by the OpenAlex API.

        Returns
        -------
        List[str]
            List of author display names.

        Notes
        -----
        If no valid author information is present, a placeholder author
        value is returned.
        """
        if not authorships:
            return ["No author found"]
        names = []
        for a in authorships:
            author = a.get("author", {}) or {}
            display = author.get("display_name")
            if display:
                names.append(display)
        return names or None

    def search_openalex(self, num_results: int = 10) -> pd.DataFrame:
        """
        Search OpenAlex for publications matching stored search strings.

        Each search string is submitted to the OpenAlex API and the returned
        records are normalized and merged into the corpus state.

        Parameters
        ----------
        num_results : int, default=10
            Maximum number of results returned per search string. OpenAlex
            supports up to 200 results per query.

        Returns
        -------
        pd.DataFrame
            Updated `CorpusState.insights` DataFrame containing retrieved
            OpenAlex records.

        Retrieved Fields
        ----------------
        The following metadata fields are extracted:

            - paper_title
            - paper_author
            - paper_date
            - doi
            - search_string
            - search_engine
            - paper_id

        Notes
        -----
        Requests are retried once when rate limiting (HTTP 429) or service
        errors (HTTP 503) occur.

        Results from OpenAlex are merged with any existing literature
        records retrieved from other sources such as Crossref.
        """
        if num_results > 200:
            raise ValueError("num_results cannot exceed 200")

        output: Dict[str, List] = {
            "search_string": [],
            "paper_title": [],
            "paper_author": [],
            "paper_date": [],
            "doi": [],
        }

        session = requests.Session()
        session.headers.update({
            "User-Agent": f"lit-review-machine/1 (mailto:{os.getenv('EMAIL_DOMAIN','noreply@example.com')})"
        })

        for search_string in self.corpus_state.insights["search_string"]:
            params = {"search": search_string, "per_page": num_results}
            try:
                r = session.get(self.OPENALEX_BASE, params=params, timeout=self.OPENALEX_TIMEOUT)
                if r.status_code in (429, 503):
                    delay = 1 + random.random()
                    print(f"OpenAlex {r.status_code} — retrying in {delay:.1f}s…")
                    time.sleep(delay)
                    r = session.get(self.OPENALEX_BASE, params=params, timeout=self.OPENALEX_TIMEOUT)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"OpenAlex request failed for '{search_string}': {e}")
                continue

            data = r.json() or {}
            for item in data.get("results", []):
                # Title
                title = item.get("title") or "No title found"
                output["paper_title"].append(title)

                # Authors
                output["paper_author"].append(self._openalex_authors(item.get("authorships")))

                # Publication year
                year = item.get("publication_year")
                output["paper_date"].append(year if isinstance(year, int) else pd.NA)

                # DOI
                doi = item.get("doi")
                if isinstance(doi, str) and doi.lower().startswith("https://doi.org/"):
                    doi = doi.split("https://doi.org/", 1)[1]
                output["doi"].append(doi or pd.NA)

                # Search string
                output["search_string"].append(search_string)

        df = pd.DataFrame(output)
        df["search_engine"] = "OpenAlex"
        df["paper_id"] = [f"openalex_paper_{i+1}" for i in range(len(df))]
        df["paper_author"] = [";".join(authors) if isinstance(authors, list) else authors for authors in df["paper_author"]]

        merged = self._merge_search_results_with_state(df)
        merged = merged.drop_duplicates(
            subset=["paper_title", "paper_date", "search_engine", "search_string"]
        ).reset_index(drop=True)

        self.corpus_state.insights = merged
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "02_academic_lit"))
        return self.corpus_state.insights
    
class GreyLiterature:
    """
    Retrieve and manage grey literature using LLM-assisted web search.

    This class identifies grey literature relevant to research questions
    by using an LLM with web search capability. Grey literature is defined
    broadly as non–peer-reviewed research outputs such as:

        - policy reports
        - working papers
        - think-tank publications
        - NGO or multilateral reports
        - institutional case studies

    Retrieved records are normalized and merged into `CorpusState.insights`
    so they can be used alongside academic literature in the ReadingMachine
    pipeline.

    Workflow
    --------
    1. Initialize with a CorpusState or a list of research questions.
    2. Construct a prompt containing the research questions.
    3. Call a reasoning-capable LLM with web search enabled.
    4. Clean the returned JSON using a secondary chat-completion model.
    5. Convert results to a structured DataFrame.
    6. Merge results into `CorpusState.insights`.
    7. Persist results to disk.

    Notes
    -----
    This class is part of the `getlit` discovery toolkit rather than the
    core ReadingMachine analytical pipeline. It is intended to assist
    users in identifying candidate documents to include in their corpus.
    """

    # Default path for caching grey literature results
    def __init__(
        self,
        llm_client: Any,  # Client interface for interacting with the LLM API
        corpus_state: Optional["CorpusState"] = None,  # Current research corpus_state (can be injected)
        questions: Optional[List[str]] = None,    # User-defined research questions
        ai_reasoning_model: str = "o3-deep-research",       # LLM model to use
        ai_chat_completion_model: str = "gpt-4o",  # Chat completion model for JSON cleaning
        grey_lit_pickle_folder: str = os.path.join(os.getcwd(), "data", "pickles") # The pickle location for the valid processed json response from the LLM
        
        #GREY_LIT_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit.pkl"), # The pickle location for the valid processed json response from the LLM
        #GREY_LIT_RAW_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit_raw.pkl") # If the LLM fails to return a valid json the raw output gets saved here - as this is an expensive call
    ) -> None:
        """
        Initialize the GreyLiterature retrieval tool.

        Parameters
        ----------
        llm_client : Any
            Client used to communicate with LLM APIs.

        corpus_state : Optional[CorpusState]
            Existing corpus state containing search strings and literature
            records. If provided, grey literature results will be merged into
            this state.

        questions : Optional[List[str]]
            List of research questions used to generate grey literature
            search prompts. If provided, they will be converted into the
            minimal format required for a `CorpusState`.

        ai_reasoning_model : str, default="o3-deep-research"
            Reasoning-capable LLM model used for web-based grey literature
            discovery.

        ai_chat_completion_model : str, default="gpt-4o"
            Chat completion model used to clean and validate the JSON output
            returned by the reasoning model.

        grey_lit_pickle_folder : str
            Directory used to cache grey literature retrieval results so
            expensive web-search calls do not need to be repeated.
        """

        # If questions are provided directly, format them into a DataFrame
        if questions:
            question_id: List[str] = [f"Question_{i}" for i in range(len(questions))]
            questions = pd.DataFrame({
                "question_id": question_id,
                "question_text": questions
            })

        # Validate the corpus_state and inject the questions if provided
        self.corpus_state: "CorpusState" = deepcopy(
            utils.validate_format(
                corpus_state=corpus_state,
                injected_value=questions,
                state_required_cols=[
                    "question_id", "question_text", "search_string_id", "search_string",
                    "search_engine","paper_id", "paper_title", "paper_author", "paper_date", "doi"
                    ],
                injected_required_cols=["question_id", "question_text"]
                )
        )

        self.llm_client: Any = llm_client
        self.ai_reasoning_model: str = ai_reasoning_model
        self.ai_chat_completion_model: str = ai_chat_completion_model
        self.grey_lit_pickle_folder = grey_lit_pickle_folder
        #self.GREY_LIT_PICKLE_FILE = os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl")
        #self.GREY_LIT_RAW_PICKLE_FILE = os.path.join(self.grey_lit_pickle_folder, "grey_lit_raw.pkl")


    def get_grey_lit(self, example_grey_literature_sources, resp_timeout=1500) -> Optional[pd.DataFrame]:
        """
        Retrieve grey literature relevant to the research questions.

        This method uses an LLM with web search capability to identify
        grey literature sources relevant to the research questions stored
        in the current `CorpusState`.

        The retrieval process proceeds through several stages:

            1. Build a prompt containing the research questions.
            2. Call a reasoning-capable LLM with web search enabled.
            3. Parse and clean the returned JSON using a chat completion model.
            4. Normalize the results into a structured DataFrame.
            5. Merge results with the existing corpus state.
            6. Persist the updated corpus state and cache the results.

        Parameters
        ----------
        example_grey_literature_sources : list
            Example institutions or sources that publish relevant grey
            literature. These are used to guide the LLM's search.

        resp_timeout : int, default=1500
            Maximum allowed time (seconds) for the reasoning model call.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing retrieved grey literature records or
            None if retrieval fails.

        Returned Data Columns
        ---------------------
        - question_id
        - question_text
        - paper_id
        - paper_title
        - paper_author
        - paper_date
        - doi

        Notes
        -----
        Results are cached to disk to avoid repeating expensive web-search
        LLM calls. If cached results exist, the user can choose to recover
        them instead of running the search again.
        """

        if os.path.exists(os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl")):
            recover = None
            while recover not in ["r", "n"]:
                recover = input("AI generated grey literature already exists. Would you like to recover (r) or generate new (n)? (r/n): ").lower()
            if recover == "r":
                with open(os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl"), "rb") as f:
                    self.grey_lit = pickle.load(f)

                self.corpus_state.insights = pd.concat([self.corpus_state.insights, self.grey_lit], ignore_index=True)
                return self.grey_lit

        # Build question strings: "question_id: question_text"
        question_strings = (
            self.corpus_state.insights[["question_id", "question_text"]]
            .drop_duplicates()
            .assign(combined=lambda df: df["question_id"].astype(str) + ": " + df["question_text"])
            ["combined"]
            .to_list()
        )

        # Build LLM prompt for the reasoning model
        prompt: str = Prompts().grey_lit_retrieve(questions=question_strings, 
                                                  example_grey_literature_sources=example_grey_literature_sources)


        # Call the LLM
        response_dict = utils.call_reasoning_model(prompt=prompt,
                                        llm_client=self.llm_client,
                                        ai_model=self.ai_reasoning_model,
                                        resp_timeout=resp_timeout                                            
                                        )
        
        
        if response_dict["status"] == "success":
            response = response_dict["response"]
            self.raw_grey_lit_response = response
        else:
            print("LLM call for grey literature completed but did not return output_text full trace is being returned.")
            return response_dict["response"]

        # Have a chat completion model clean the json
        # First create the fall back reflecting the json structure i am asking for back from the LLM
        fallback = {
            "results": [
                {
                    "question_id": "",
                    "paper_title": "",
                    "paper_author": "",
                    "paper_date": "",
                    "doi": None
                }
            ]
        }
        print("Passing the result of the AI assisted search to an LLM for cleaning....")
        # Now call the chat completion model to clean the JSON
        clean_response = utils.call_chat_completion(ai_model=self.ai_chat_completion_model, 
                                              llm_client=self.llm_client,
                                              sys_prompt=Prompts().grey_literature_format_check(),
                                              user_prompt=response,
                                              return_json=True,
                                              fall_back=fallback)
        
        self.clean_grey_lit_response = clean_response
        

        # Get grey lit from the json and convert to dataframe
        grey_lit_list = clean_response["results"]
        grey_lit_pd = pd.DataFrame(grey_lit_list)

        # Prefix paper_id with "grey_lit_"
        grey_lit_pd["paper_id"] = [f"grey_lit_{i}" for i in range(len(grey_lit_pd))]

        # Merge ONLY on canonical `question_id` to get original question_text
        grey_lit_pd = grey_lit_pd.merge(
            self.corpus_state.insights[["question_id", "question_text"]].drop_duplicates(),
            on="question_id",
            how="left"
        )

        grey_lit_pd = grey_lit_pd.replace(["", "NA", pd.NA, np.nan, "null"], None)

        # Create grey lit attribute for inspection 
        self.grey_lit = grey_lit_pd

        # Pickle the result so that it can be accessed later - this is an expensive call
        os.makedirs(self.grey_lit_pickle_folder, exist_ok=True)
        # Atomic pickle with safe pickle utility function to avoid corrupting the cache with failed writes
        utils.safe_pickle(self.grey_lit, os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl"))

        # Update corpus_state
        self.corpus_state.insights = pd.concat([self.corpus_state.insights, self.grey_lit], ignore_index=True)
        self.corpus_state.insights["paper_date"] = pd.to_numeric(self.corpus_state.insights["paper_date"], errors="coerce").astype("Int64")
        self.corpus_state.insights.replace(["", "NA", pd.NA, np.nan, "null"], None, inplace=True)
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "03_grey_lit"))

        # Return grey literature
        return self.grey_lit

class Literature:
    """
    Manage and deduplicate retrieved literature records.

    This class identifies and resolves duplicate literature records
    retrieved from multiple sources (e.g., Crossref, OpenAlex, grey literature).

    Deduplication occurs in two stages:

        1. Exact duplicate removal
        2. Fuzzy duplicate detection (human-in-the-loop)

    Deduplication is GLOBAL across all questions to avoid redundant ingestion.

    Workflow
    --------
    1. Remove exact duplicates
    2. Generate fuzzy match candidates
    3. Export for manual review
    4. User deletes duplicates
    5. Reload cleaned dataset into CorpusState
    """

    def __init__(
        self,
        corpus_state: "CorpusState",
        literature: Optional[pd.DataFrame] = None,
        fuzzy_check_path: str = config.FUZZY_CHECK_PATH,
    ) -> None:

        self.RUN = "getlit"
        self.fuzzy_check_path = fuzzy_check_path
        self.save_location = os.path.join(self.fuzzy_check_path, self.RUN)

        os.makedirs(self.save_location, exist_ok=True)

        self.corpus_state: "CorpusState" = deepcopy(
            utils.validate_format(
                corpus_state=corpus_state,
                injected_value=literature,
                state_required_cols=[
                    "question_id", "question_text", "search_string_id", "search_string",
                    "paper_id", "paper_title", "paper_author", "paper_date",
                    "doi"
                ],
                injected_required_cols=[
                    "question_id", "paper_id", "paper_title", "paper_author",
                    "paper_date", "doi"
                ]
            )
        )

    def drop_duplicates(self) -> pd.DataFrame:
        """
        Remove exact duplicates from corpus_state.

        Returns
        -------
        pd.DataFrame
            Deduplicated insights DataFrame.
        """

        unique_df = utils.drop_exact_duplicates(self.corpus_state.insights)

        # Update state
        self.corpus_state.insights = unique_df

        return unique_df

    def get_fuzzy_matches(self, similarity_threshold: int = 90) -> None:
        """
        Generate and export fuzzy duplicate candidates for manual review.

        Parameters
        ----------
        similarity_threshold : int
            Threshold for fuzzy matching.

        Returns
        -------
        None
        """

        review_df = utils.prepare_fuzzy_review_df(
            self.corpus_state.insights,
            similarity_threshold=similarity_threshold
        )

        output_path = os.path.join(self.save_location, "fuzzy_matches.csv")
        review_df.to_csv(output_path, index=False)

        print(
            f"Fuzzy matches exported to {output_path}.\n"
            "Review the file and DELETE duplicate rows.\n"
            "Save as CSV and run update_state().\n\n"
            "NOTE: Deduplication is GLOBAL — later questions may show fewer papers.\n"
            "This is expected and ensures no duplicate ingestion."
        )

        return None

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
        Update corpus state using manually reviewed duplicate file.

        At this point in the pipeline we are just loading the deduped insights back into state. 
        So there isnt any processing to do, other than to drop the helper columns - done by leaving them out of output_cols.
        ----------
        filename : str
            Name of the manually reviewed file.

        encoding : str, default="utf-8"
            Encoding of the manually reviewed file.

        output_cols : list
            List of columns to include in the updated insights DataFrame. Helper columns used for review should be excluded from this list.

        Returns
        -------
        pd.DataFrame
            Cleaned insights DataFrame.
        """
        # Set the file path for the manually reviewed file
        filepath = os.path.join(self.save_location, filename)

        # Load manually reviewed file, ensuring only the specified columns are included
        insights_df = CorpusState.load_insights_from_csv_xslx(filepath=filepath, encoding=encoding, output_cols=output_cols)

        # Assign to state
        self.corpus_state.insights = insights_df

        # Save
        self.corpus_state.save(
            os.path.join(config.STATE_SAVE_LOCATION, "04_literature_deduped")
        )

        # Return the cleaned insights for inspection if needed
        return self.corpus_state.insights


class AiLiteratureCheck:
    """
    Identify potentially missing literature using an LLM.

    This class performs a literature completeness check by asking an LLM
    to review the current corpus of retrieved publications and suggest
    additional papers that may be relevant to the research questions.

    The LLM receives a structured JSON representation of the currently
    identified literature and returns candidate missing papers for each
    research question.

    Suggested papers are normalized and merged into `CorpusState.insights`
    so that they can be reviewed and potentially included in the corpus.

    Workflow
    --------
    1. Convert the existing literature corpus into structured JSON.
    2. Send the JSON to an LLM capable of reasoning and web search.
    3. Parse the returned JSON containing suggested papers.
    4. Clean and normalize the returned results.
    5. Merge the suggested papers into the corpus state.

    Notes
    -----
    This class is part of the `getlit` discovery toolkit and is intended
    as a **sanity check** for literature coverage. It helps users identify
    potential gaps in the collected literature but does not automatically
    retrieve full documents.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_reasoning_model: str = "o3-deep-research",
        ai_chat_completion_model: str = "gpt-4o",
        corpus_state: Optional["CorpusState"] = None,
        papers: Optional[pd.DataFrame] = None, 
    ) -> None:
        """
        Initialize the AI literature completeness checker.

        Parameters
        ----------
        llm_client : Any
            Client used to communicate with the LLM API.

        ai_reasoning_model : str, default="o3-deep-research"
            Reasoning-capable model used to identify missing literature.

        ai_chat_completion_model : str, default="gpt-4o"
            Chat completion model used to clean and validate the JSON
            returned by the reasoning model.

        corpus_state : Optional[CorpusState]
            Existing pipeline state containing literature records.

        papers : Optional[pd.DataFrame]
            Optional DataFrame containing literature records to inject
            if a corpus state is not provided.

        Notes
        -----
        The constructor validates the input data to ensure that the
        literature records contain the required metadata fields.
        """
        self.llm_client: Any = llm_client
        self.ai_reasoning_model: str = ai_reasoning_model
        self.ai_chat_completion_model: str = ai_chat_completion_model

        # Validate that the corpus_state or injected papers contain all required columns
        self.corpus_state: "CorpusState" = deepcopy(
            utils.validate_format(
            corpus_state=corpus_state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "doi"
            ],
            injected_required_cols=[
                "question_id", "question_text", "paper_id", "paper_title", "paper_author",
                "paper_date", "doi"
            ]
            )
        )

        # Preprocess current literature into JSON for LLM prompt insertion
        self.json_for_prompt_insertion: str = self._clean_data_for_prompt_insertion()

    def _clean_data_for_prompt_insertion(self) -> str:
        """
        Convert literature records into JSON for LLM prompt insertion.

        This method prepares the current literature corpus in a structured
        JSON format suitable for LLM processing. Papers are grouped by
        research question so the model can evaluate literature coverage
        within the context of each question.

        Returns
        -------
        str
            JSON string containing grouped literature records in the form:

                [
                    {
                        "question_id": "...",
                        "question_text": "...",
                        "papers": [...]
                    }
                ]

        Notes
        -----
        The resulting JSON is embedded directly into the prompt used for
        the AI literature completeness check.
        """
        df = self.corpus_state.insights[[
            "question_id", "question_text", "paper_id", "paper_author", "paper_date", "paper_title"
        ]]

        json_list = [
            {
                "question_id": qid,
                "question_text": qtext,
                "papers": group[["paper_id", "paper_author", "paper_date", "paper_title"]]
                        .to_dict(orient="records")
            }
            for (qid, qtext), group in df.groupby(["question_id", "question_text"], sort=False)
            ]

        return json.dumps(json_list, indent=2)

    def ai_literature_check(self, resp_timeout = 1500) -> Optional[pd.DataFrame]:
        """
        Identify potentially missing literature using an LLM.

        This method sends the current literature corpus to a reasoning-capable
        language model and asks it to identify additional papers that may be
        relevant to each research question.

        The returned suggestions are cleaned, normalized, and merged into
        the corpus state.

        Parameters
        ----------
        resp_timeout : int, default=1500
            Maximum allowed time (seconds) for the reasoning model request.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing the AI-suggested literature records.

            Columns include:

                - question_id
                - question_text
                - paper_id
                - paper_title
                - paper_author
                - paper_date
                - doi

            If no additional papers are suggested, the existing corpus state
            is returned.

        Workflow
        --------
        1. Generate a prompt containing the current literature corpus.
        2. Send the prompt to a reasoning-capable LLM.
        3. Clean the returned JSON using a secondary chat completion model.
        4. Normalize the results into a DataFrame.
        5. Merge the results into `CorpusState.insights`.
        6. Persist the updated state to disk.

        Notes
        -----
        AI-suggested papers receive unique identifiers prefixed with
        `"ai_lit_"` to distinguish them from papers retrieved through
        other discovery methods.
        """
        # Generate the prompt using preprocessed JSON
        prompt: str = Prompts().ai_literature_retrieve(
            questions_papers_json=self.json_for_prompt_insertion
        )

         # Call the LLM
        response_dict = utils.call_reasoning_model(prompt=prompt,
                                        llm_client=self.llm_client,
                                        ai_model=self.ai_reasoning_model,
                                        resp_timeout=resp_timeout                                            
                                        )
        
        
        if response_dict["status"] == "success":
            response = response_dict["response"]
            self.raw_ai_lit_check = response
        else:
            print("LLM call for grey literature completed but did not return output_text full trace is being returned.")
            return response_dict["response"]

        # Have a chat completion model clean the json
        # First create the fall back reflecting the json structure i am asking for back from the LLM
        fallback = {
            "results": [
                {
                    "question_id": "",
                    "paper_title": "",
                    "paper_author": "",
                    "paper_date": "",
                    "doi": None
                }
            ]
        }
        print("Passing the result of the AI assisted search to an LLM for cleaning....")
        # Now call the chat completion model to clean the JSON
        clean_response = utils.call_chat_completion(ai_model=self.ai_chat_completion_model, 
                                              llm_client=self.llm_client,
                                              sys_prompt=Prompts().ai_literature_format_check(),
                                              user_prompt=response,
                                              return_json=True,
                                              fall_back=fallback)
        
        self.clean_ai_lit_check = clean_response
        
        # if the response is empty no new papers were found so we print a message and save the current corpus_state and return the corpus_state.insights
        if len(clean_response["results"]) == 0:
            print("No missing papers returned by the LLM.")
            self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "05_ai_lit_check"))
            return self.corpus_state.insights

        # Otherwise we convert to a df, clean and concat with corpus_state.insights
        ai_lit = pd.DataFrame(clean_response["results"])
        # Clean:
        # Get canonical question text and join
        canonical_questions = self.corpus_state.insights[["question_id", "question_text"]].drop_duplicates()
        ai_lit = ai_lit.merge(
            canonical_questions,
            how="left",
            on="question_id"
        )
        # Assign unique AI paper IDs
        ai_lit["paper_id"] = [f"ai_lit_{i}" for i in range(ai_lit.shape[0])]
        
        # Create ai_lit attribute for inspection
        self.ai_lit = ai_lit

        # Append AI literature to corpus_state
        updated_state = pd.concat([self.corpus_state.insights, ai_lit], ignore_index=True)
        # CLean up any dates to numeric to not break parquet
        updated_state["paper_date"] = pd.to_numeric(updated_state["paper_date"], errors="coerce").astype("Int64") 
        # Clean up any empty strings to None to not break parquet
        updated_state.replace(["", "NA", pd.NA, np.nan, "null", "No author found"], None, inplace=True)
        # Update the oder for pretty export
        updated_state = updated_state.reindex(columns=[
            "question_id",
            "question_text",
            "search_string_id",
            "search_string",
            "paper_id",
            "paper_title",
            "paper_author",
            "paper_date",
            "doi"
            ])

        # Asign to corpus_state attribute
        self.corpus_state.insights = updated_state

        # Save
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "05_ai_lit_check"))

        # Return only the new AI-suggested papers
        return self.ai_lit

class DownloadManager:
    """
     Manage manual downloading of literature identified during corpus discovery.

    This class prepares a filesystem structure for storing documents
    associated with literature records in `CorpusState`. It is designed
    for workflows where documents must be downloaded manually from
    external sources.

    The class performs the following tasks:

        1. Validate literature records stored in `CorpusState`.
        2. Create a folder structure organized by research question.
        3. Export literature metadata to CSV for manual download tracking.
        4. Provide a mechanism to reload updated download metadata.

    Downloaded files are expected to be placed into folders named after
    the corresponding `question_id`. File names should ideally match
    their `paper_id` to preserve traceability between the document
    files and the literature records.

    Notes
    -----
    This class does not perform automated downloading. It provides a
    structured workflow for managing manual downloads while preserving
    the pipeline's traceability between literature records and
    documents.
    """

    def __init__(
        self,
        corpus_state: "CorpusState" = None,
        papers: Optional[pd.DataFrame] = None,
        DOWNLOAD_LOCATION: str = os.path.join(os.getcwd(), config.CORPUS_LOCATION)
    ) -> None:
        """
        Initialize the DownloadManager.

        Parameters
        ----------
        corpus_state : CorpusState, optional
            Existing pipeline state containing literature records.

        papers : Optional[pd.DataFrame]
            DataFrame containing literature records to inject if a
            corpus state is not provided.

        DOWNLOAD_LOCATION : str
            Directory where downloaded documents should be stored.

        Notes
        -----
        During initialization the following actions occur:

            - Validate the input literature records.
            - Ensure a download directory exists.
            - Create subfolders for each research question.
            - Export literature metadata to a CSV file so users can track
            download status manually.
            - Sanitize identifiers used for filenames to ensure they are
            compatible with the filesystem.
        """
        # Validate that the corpus_state or injected papers contain all required columns
        self.corpus_state: "CorpusState" = deepcopy(
            utils.validate_format(
            corpus_state=corpus_state,
            injected_value=papers,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi"
            ],
            injected_required_cols=[
                "question_id", "question_text",
                "paper_id", "paper_title", "paper_author", "paper_date",
                "doi"
            ]
            )
        )

        # Check if download_status variable is in the passed corpus_state if its not create with 0 values, assuming no downloads have happened yet, if it does exsist, simply use it. 
        if "download_status" not in self.corpus_state.insights.columns:
            self.corpus_state.insights["download_status"] = 0
        else: 
            pass

        # Ensure the base download folder exists
        self.DOWNLOAD_LOCATION: str = DOWNLOAD_LOCATION
        os.makedirs(self.DOWNLOAD_LOCATION, exist_ok=True)

        # Create subfolders for each question_id
        self._create_download_folder()

        # Preserve original IDs and sanitize for filesystem-safe filenames
        self.corpus_state.insights["messy_question_id"] = self.corpus_state.insights["question_id"]
        self.corpus_state.insights["messy_paper_id"] = self.corpus_state.insights["paper_id"]
        self.corpus_state.insights["question_id"] = self.corpus_state.insights["question_id"].apply(self._sanitize_filename)
        self.corpus_state.insights["paper_id"] = self.corpus_state.insights["paper_id"].apply(self._sanitize_filename)

        # write the insights to csv
        self.corpus_state.write_to_csv(save_location= self.DOWNLOAD_LOCATION, 
                                write_full_text=False, write_chunks=False)
        
        print(
            f"Architecture for downloading papers has been created at {self.DOWNLOAD_LOCATION}.\n"
           f"You should manually download files and update their status in the file at {os.path.join(self.DOWNLOAD_LOCATION, 'insights.csv')}. "
            "Assuming you do not change the files location the easiest way to do this is to call DownloadManager.update()\n"
            f"Note should you want to link papers to the search terms that generated them:\n"
            "1. You must save them in the folder corresponding to their question ID.\n" 
            "2. You must ensure the filenames match the paper_id in the form paper_id.[relevant extension].\n\n"
            "Allocating to folders and matching filenames is not necessary. All documents in data/corpus/ will be processed (after additional deduplication)."
            )

    def _create_download_folder(self) -> None:
        """
        Create directory structure for manual downloads.

        A separate folder is created for each `question_id`. This allows
        downloaded documents to be organized according to the research
        question they address.

        The resulting directory structure resembles:

            data/corpus/
                question_0/
                question_1/
                question_2/
        """
        for qid in self.corpus_state.insights["question_id"].unique():
            os.makedirs(os.path.join(self.DOWNLOAD_LOCATION, qid), exist_ok=True)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Convert an identifier into a filesystem-safe filename.

        Illegal characters used in file paths are replaced with
        underscores so that question IDs and paper IDs can safely be
        used as folder names or filenames.

        Parameters
        ----------
        filename : str
            Original identifier string.

        Returns
        -------
        str
            Sanitized filename safe for filesystem use.
        """
        sanitized = re.sub(r'[\\/:*?"<>|]', "_", filename)
        return sanitized.strip()
    
    def update_state(
            self, 
            filename: str, 
            encoding: str = "utf-8", 
            output_cols: List = [
                "question_id",
                "question_text",
                "search_string_id",
                "search_string",
                "paper_id",
                "paper_title",
                "paper_author",
                "paper_date",
                "doi",
                "download_status",
                "messy_question_id",
                "messy_paper_id"
                ]
        ) -> pd.DataFrame:
                
        """
        Reload updated download metadata into the corpus state.

        This method reads the CSV files stored in the download directory
        and updates the `CorpusState` with any manual changes made to
        the literature metadata (for example updated download status).

        At this point in the pipeline we are just loading the 

        Returns
        -------
        pd.DataFrame
            Updated `CorpusState.insights` DataFrame.

        Notes
        -----
        This method is used after the user has manually
        downloaded documents.
        """
        # Set the filepaath for the manually reviewed file
        filepath = os.path.join(self.DOWNLOAD_LOCATION, filename)
        # Call the convennience function to load the insights from csv/xlsx with the schema matching what it should look like at this point in the pipeline. 
        insights_df = CorpusState.load_insights_from_csv_xslx(filepath=filepath, encoding=encoding, output_cols=output_cols)
        # Update state object with the loaded insights
        self.corpus_state.insights = insights_df
        # Save
        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "06_download_manager"))
        # Return for inspection
        return(self.corpus_state.insights)
