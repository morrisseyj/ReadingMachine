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
from rapidfuzz import process, fuzz
import networkx as nx
import re
from kneed import KneeLocator



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
    
# class DOI:
#     """
#     Retrieve DOIs and open-access download links for papers stored in a CorpusState
#     object or provided as a DataFrame.
#     """

#     def __init__(
#         self, 
#         corpus_state: Optional["CorpusState"] = None, 
#         papers: Optional[pd.DataFrame] = None
#     ) -> None:
#         """
#         Initialize DOI retriever.

#         Args:
#             corpus_state: A pre-existing CorpusState object containing paper metadata.
#             papers: A DataFrame containing paper metadata (used if corpus_state is None).
#         """
#         # Validate and set up corpus_state
#         self.corpus_state = deepcopy(
#             validate_format(
#             corpus_state=corpus_state,
#             injected_value=papers,
#             state_required_cols=[
#                 "question_id", "question_text", "search_string_id", "search_string", 
#                 "search_engine", "doi", "paper_id", "paper_title", "paper_author", "paper_date"
#             ],
#             injected_required_cols=[
#                 "question_id", "question_text", "paper_id", 
#                 "paper_title", "paper_author", "paper_date"
#             ]
#             )
#             )

#         # Ensure folder exists for pickle
#         os.makedirs(os.path.dirname(STATE_SAVE_LOCATION), exist_ok=True)

#         # Store search strings for DOI lookup``
#         self.search_string: List[str] = self._create_search_string()

#     def _create_search_string(self) -> List[str]:
#         """
#         Concatenates paper title, authors, and year into search strings for DOI lookups.

#         Returns:
#             List[str]: A list of search strings for each paper.
#         """
#         if self.corpus_state.insights.empty:
#             return []

#         df = self.corpus_state.insights[["paper_title", "paper_author", "paper_date"]].copy()
#         df["search_string"] = (
#         df["paper_title"].astype(str) + " " +
#         df["paper_author"].astype(str) + " " +
#         df["paper_date"].astype(str)
#         )

#         search_string = df["search_string"].tolist()
#         return search_string

#     @staticmethod
#     def call_alex(search_string: str) -> Optional[str]:
#         """
#         Queries the OpenAlex API with a search string to retrieve a DOI.

#         Args:
#             search_string: A string composed of title, author(s), and year.

#         Returns:
#             Optional[str]: The DOI string if found, otherwise None.
#         """
#         url = "https://api.openalex.org/works"
#         params = {"search": search_string, "per-page": 1}
#         try:
#             response = requests.get(url, params=params, timeout=10)
#             if response.status_code == 200:
#                 items = response.json().get("results", [])
#                 if items:
#                     doi_url = items[0].get("doi")
#                     if doi_url:
#                         return doi_url.removeprefix("https://doi.org/")
#         except requests.exceptions.RequestException:
#             pass

#         time.sleep(1)  # Prevent hitting API rate limits
#         return None

#     def get_doi(self) -> List[Optional[str]]:
#         """
#         Retrieves DOIs for all papers in the current corpus_state using OpenAlex.

#         Returns:
#             List[Optional[str]]: A list of DOIs corresponding to papers.
#         """
#         dois = self.corpus_state.insights["doi"]

#         if not self.search_string:
#             print("No papers available to retrieve DOIs.")
#             self.corpus_state.insights["doi"] = []
#             return []

#         for idx, (string, doi) in enumerate(zip(self.search_string, dois), start=1):
#             print(f"Retrieving DOI {idx} of {len(self.search_string)}")
#             if not pd.isna(doi):
#                 continue  # Skip if DOI already exists from the AcademicLit search
#             else:
#                 doi_result = self.call_alex(string)
#                 dois[idx - 1] = doi_result

#         self.corpus_state.insights["doi"] = dois

#         return dois

#     def get_download_link(self) -> List[Optional[str]]:
#         """
#         Retrieves open-access PDF download links for each paper via Unpywall.

#         Returns:
#             List[Optional[str]]: A list of open-access PDF download links (or None if unavailable).
#         """
#         # Ensure DOI column exists
#         self.corpus_state = validate_format(
#             corpus_state=self.corpus_state,
#             injected_value=self.corpus_state.insights,
#             state_required_cols=[
#                 "question_id", "question_text", "search_string_id", "search_string",
#                 "search_engine","paper_id", "paper_title", "paper_author", "paper_date", "doi"
#             ],
#             injected_required_cols=[
#                 "question_id", "question_text", "paper_id",
#                 "paper_title", "paper_author", "paper_date", "doi"
#             ]
#         )

#         download_links: List[Optional[str]] = []

#         for idx, doi in enumerate(self.corpus_state.insights.get("doi", []), start=1):
#             print(f"Retrieving downlod link for paper {idx} of {self.corpus_state.insights.shape[0]}")

#             if not doi:
#                 download_links.append(None)
#                 continue

#             try:
#                 unpay = Unpywall.get_json(doi)
#                 oa_locations = unpay.get("oa_locations", [])
#                 if not oa_locations:
#                     download_links.append(None)
#                     continue
#             except Exception as e:
#                 download_links.append(f"Error: {e}")
#                 continue

#             for loc in oa_locations:
#                 url = loc.get("url_for_pdf")
#                 if url:
#                     try:
#                         response = requests.head(url, allow_redirects=True, timeout=10)
#                         if (
#                             response.status_code == 200
#                             and "application/pdf" in response.headers.get("Content-Type", "")
#                         ):
#                             download_links.append(url)
#                             break
#                     except requests.exceptions.RequestException:
#                         continue
#             else:
#                 download_links.append(None)

#         self.corpus_state.insights["download_link"] = download_links
#         self.corpus_state.save(STATE_SAVE_LOCATION)

#         return download_links

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
        with open(os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl"), "wb") as f:
            pickle.dump(self.grey_lit, f)

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

    This class provides utilities for identifying and resolving
    duplicate literature records retrieved from multiple sources
    (e.g., Crossref, OpenAlex, grey literature retrieval).

    The deduplication process occurs in two stages:

        1. Exact duplicate removal
           Records with identical author–title–year combinations are
           automatically removed.

        2. Fuzzy duplicate detection
           Records with highly similar author–title–year strings are
           flagged as potential duplicates for manual inspection.

    The workflow supports human-in-the-loop validation by exporting
    candidate duplicates to CSV files for manual review before updating
    the pipeline state.

    Workflow
    --------
    1. Split literature records by research question.
    2. Construct a string used for duplicate detection.
    3. Remove exact duplicates.
    4. Detect fuzzy duplicates using pairwise string similarity.
    5. Export potential duplicates for manual review.
    6. Import manually reviewed files and update `CorpusState`.

    Notes
    -----
    Deduplication occurs per research question to prevent unrelated
    questions from influencing duplicate detection.
    """

    FUZZY_CHECK_PATH: str = os.path.join(os.getcwd(), "data", "fuzzy_check")
    os.makedirs(FUZZY_CHECK_PATH, exist_ok=True)

    def __init__(self, corpus_state: "CorpusState", literature: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the Literature deduplication tool.

        Parameters
        ----------
        corpus_state : CorpusState
            Pipeline state containing literature records retrieved from
            academic and grey literature search tools.

        literature : Optional[pd.DataFrame]
            Optional DataFrame containing literature records to be injected
            into the corpus state.

        Notes
        -----
        Input data is validated to ensure the required columns exist before
        deduplication operations are performed.
        """
        
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

        # Split literature into a list of DataFrames per question_id
        self.question_dfs: List[pd.DataFrame] = self._splitter()

    def _splitter(self) -> List[pd.DataFrame]:
        """
        Split literature records by research question.

        Each research question is processed independently so that duplicate
        detection occurs within the context of that question's literature.

        The method also constructs a concatenated string combining:

            - author names
            - paper title
            - publication year

        This string is used for both exact and fuzzy duplicate detection.

        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames, one per research question.
        """
        dfs: List[pd.DataFrame] = [
            self.corpus_state.insights[self.corpus_state.insights["question_id"] == qid].copy()
            for qid in self.corpus_state.insights["question_id"].drop_duplicates()
        ]

        for df in dfs:
            authors_str = df["paper_author"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            title_str = df["paper_title"].astype(str)
            date_str = df["paper_date"].astype(str)

            # Concatenate for duplicate checking
            df["duplicate_check_string"] = authors_str + " " + title_str + " " + date_str

        return dfs

    def drop_exact_duplicates(self) -> List[pd.DataFrame]:
        """
        Remove exact duplicate literature records.

        Exact duplicates are identified using the constructed
        `duplicate_check_string`, which combines author names,
        paper title, and publication year.

        Returns
        -------
        List[pd.DataFrame]
            Updated list of question-specific DataFrames with exact
            duplicates removed.
        """
        for df in self.question_dfs:
            df.drop_duplicates(subset="duplicate_check_string", keep="first", inplace=True)
        return self.question_dfs

    def _get_fuzzy_match(self, similarity_threshold: int = 90) -> List[List[Tuple[str, str]]]:
        """
        Identify potential duplicate records using fuzzy string matching.

        Pairwise similarity scores are computed for all literature records
        within each research question using the RapidFuzz library.

        Parameters
        ----------
        similarity_threshold : int, default=90
            Minimum similarity score required for two records to be
            considered potential duplicates.

        Returns
        -------
        List[List[Tuple[str, str]]]
            List of lists containing pairs of potentially duplicated
            literature records for each research question.

        Notes
        -----
        The similarity score is calculated using the RapidFuzz `ratio`
        scorer, which measures overall string similarity.
        """
        fuzzy_duplicates_list: List[List[Tuple[str, str]]] = []

        for df in self.question_dfs:
            strings = df["duplicate_check_string"].tolist()
            fuzzy_scores = process.cdist(strings, strings, scorer=fuzz.ratio)
            unique_fuzzy_matches: List[Tuple[str, str]] = []

            for i, row in enumerate(fuzzy_scores):
                for j, score in enumerate(row):
                    if i < j and score >= similarity_threshold:
                        unique_fuzzy_matches.append((strings[i], strings[j]))

            fuzzy_duplicates_list.append(unique_fuzzy_matches)

        print("Pairwise fuzzy score calculated.")
        return fuzzy_duplicates_list

    def _get_similar_groups(self) -> List[pd.DataFrame]:
        """
        Group fuzzy duplicate matches into connected similarity clusters.

        Potential duplicate pairs are converted into a graph structure where
        edges represent high-similarity matches. Connected components in the
        graph represent groups of records that may correspond to the same
        publication.

        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames containing similarity group assignments
            for each research question.

        Notes
        -----
        Records without fuzzy matches are assigned `sim_group = -1`.
        """

        fuzzy_groups_list: List[pd.DataFrame] = []
        fuzzy_duplicates_list = self._get_fuzzy_match()

        for possible_duplicates, df in zip(fuzzy_duplicates_list, self.question_dfs):
            graph = nx.Graph()
            graph.add_edges_from(possible_duplicates)
            groups = list(nx.connected_components(graph))

            grouped_matches = []
            matched_strings = set()

            for i, group in enumerate(groups, start=1):
                for string in group:
                    grouped_matches.append({"duplicate_check_string": string, "sim_group": i})
                    matched_strings.add(string)

            # Assign -1 to strings with no matches
            for string in df["duplicate_check_string"]:
                if string not in matched_strings:
                    grouped_matches.append({"duplicate_check_string": string, "sim_group": -1})

            groups_df = pd.DataFrame(grouped_matches)
            fuzzy_groups_list.append(groups_df)

        return fuzzy_groups_list

    def get_fuzzy_matches(self) -> None:
        """
        Export potential duplicate records for manual review.

        For each research question, a CSV file is generated containing
        literature records grouped by similarity cluster. These files are
        intended for manual inspection to determine which records represent
        true duplicates.

        The exported files are written to `FUZZY_CHECK_PATH`.

        Notes
        -----
        After manual review and editing, the updated files can be imported
        back into the pipeline using `update_state()`.
        """
        fuzzy_groups_list = self._get_similar_groups()

        for index, (fuzzy_group_df, df) in enumerate(zip(fuzzy_groups_list, self.question_dfs)):
            df_for_manual_check = fuzzy_group_df.merge(
                df, 
                how="left",
                on="duplicate_check_string"
            )
            df_for_manual_check.to_csv(
                os.path.join(self.FUZZY_CHECK_PATH, f"question{index + 1}.csv"),
                index=False
            )

        print(
            f"All fuzzy matches exported to {self.FUZZY_CHECK_PATH}. "
            "Check and remove any true duplicates manually. "
            "Only save as .csv to ensure update_state() works correctly."
        )

    def update_state(self, path_to_files: Optional[str] = None) -> pd.DataFrame:
        """
        Update the corpus state using manually reviewed duplicate files.

        This method reads the manually edited CSV or Excel files generated
        during fuzzy duplicate inspection and updates the pipeline state
        with the cleaned literature records.

        Parameters
        ----------
        path_to_files : Optional[str]
            Directory containing the manually reviewed literature files.
            Defaults to the configured fuzzy-check directory.

        Returns
        -------
        pd.DataFrame
            Updated `CorpusState.insights` DataFrame containing deduplicated
            literature records.

        Notes
        -----
        Temporary columns used during duplicate detection
        (`duplicate_check_string`, `sim_group`) are removed before the
        updated state is saved.
        """

        path_to_files = path_to_files or self.FUZZY_CHECK_PATH
        files_to_import = [
            os.path.join(path_to_files, f)
            for f in os.listdir(path_to_files)
            if f.lower().endswith(".csv") or f.lower().endswith(".xlsx")
        ]

        dfs: List[pd.DataFrame] = []
        for file in files_to_import:
            if file.lower().endswith(".csv"):
                dfs.append(pd.read_csv(file))
            else:
                dfs.append(pd.read_excel(file))
        
        # Clean up empty authors or "no author found" entries.
        for df in dfs:
            if "paper_author" in df.columns:
                df["paper_author"].replace(
                    to_replace=["", "No author found", "NA", "null", pd.NA, np.nan],
                    value=None,
                    inplace=True
                    )
        
        self.corpus_state.insights = pd.concat(dfs, ignore_index=True)

        if "duplicate_check_string" in self.corpus_state.insights.columns:
            self.corpus_state.insights.drop(columns="duplicate_check_string", inplace=True)

        if "sim_group" in self.corpus_state.insights.columns:
            self.corpus_state.insights.drop(columns="sim_group", inplace=True)

        self.corpus_state.save(os.path.join(config.STATE_SAVE_LOCATION, "04_literature_deduped"))
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
            f"Note when saving these files you MUST SAVE THEM IN THE FOLDER CORRESPONDING TO THIER QUESTION ID. You should also ensure the filenames match the paper_id in the form paper_id.[relevant extension]. " 
            "Matching filenames with paper_ids is not neccesary but will allow you to track papers back to search prompts. You can add papers to these folders that are not in your "
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
    
    def update(self):
        """
        Reload updated download metadata into the corpus state.

        This method reads the CSV files stored in the download directory
        and updates the `CorpusState` with any manual changes made to
        the literature metadata (for example updated download status).

        Returns
        -------
        pd.DataFrame
            Updated `CorpusState.insights` DataFrame.

        Notes
        -----
        This method is typically used after the user has manually
        downloaded documents and updated the exported metadata files.
        """
        # This convenience function just calls the from csv method of the questionstate
        self.corpus_state = self.corpus_state.from_csv(filepath=os.path.join(self.DOWNLOAD_LOCATION))
        # And updates the corpus_state object on file by saving
        self.corpus_state.save(config.STATE_SAVE_LOCATION)
        return(self.corpus_state.insights)

# THIS WAS ALL TOO FRAGILE TO MAKE WORK SO I BAILED ON IT AND RESORTED TO MANUAL DOWNLOADING
#     def download_files(self) -> pd.DataFrame:
#         """
#         Attempt to download all files in the corpus_state DataFrame. Tracks download status
#         and local filenames. Updates corpus_state and writes a CSV with download results.

#         Returns:
#             DataFrame containing columns ['paper_id', 'download_status'] with updated statuses.
#         """
#         # Ensure subfolders exist
#         self._create_download_folder()

#         # Initialize download tracking columns
#         if "download_status" not in self.corpus_state.insights.columns:
#             self.corpus_state.insights["download_status"] = 0
#         if "filename" not in self.corpus_state.insights.columns:
#             self.corpus_state.insights["filename"] = np.nan

#         # Iterate through each row and attempt download
#         for idx, row in self.corpus_state.insights.iterrows():
#             url: str = row["download_link"]
#             status: int = row["download_status"]
#             qid: str = row["question_id"]
#             pid: str = row["paper_id"]

#             print(f"Downloading file {idx + 1} of {self.corpus_state.insights.shape[0]}")

#             if status == 0:
#                 if pd.notna(url) and url != "NA":
#                     try:
#                         response = requests.get(url, stream=True, timeout=10)
#                         response.raise_for_status()

#                         file_path = os.path.join(self.DOWNLOAD_LOCATION, qid, f"{pid}.pdf")
#                         with open(file_path, "wb") as f:
#                             for chunk in response.iter_content(chunk_size=8192):
#                                 f.write(chunk)

#                         self.corpus_state.insights.at[idx, "filename"] = file_path
#                         self.corpus_state.insights.at[idx, "download_status"] = 1
#                     except Exception as e:
#                         print(f"Failed to download {url}: {e}")
#                         self.corpus_state.insights.at[idx, "filename"] = np.nan
#                         self.corpus_state.insights.at[idx, "download_status"] = 0
#                 else:
#                     self.corpus_state.insights.at[idx, "filename"] = np.nan
#                     self.corpus_state.insights.at[idx, "download_status"] = 0
#             else:
#                 self.corpus_state.insights.at[idx, "download_status"] = 1

#         # Save download status CSV for inspection
#         download_status_csv = os.path.join(self.DOWNLOAD_LOCATION, "download_status.csv")
#         self.corpus_state.insights.to_csv(download_status_csv, index=False)

#         print(
#             f"Attempted downloads complete. Inspect the results here: {download_status_csv}.\n"
#             "For files that failed to download, open this CSV, update the 'download_link' as needed, and save it.\n"
#             "Then reload the updated CSV into a CorpusState using:\n"
#             "    corpus_state = CorpusState.load_from_csv('path/to/download_status.csv')\n"
#             "After that, pass the new corpus_state to the Downloader and retry downloads:\n"
#             "    downloader = Downloader(corpus_state=corpus_state)\n"
#             "Filenames correspond to sanitized question_id and paper_id, preserving traceability."
#         )
#         # Save the corpus_state
#         self.corpus_state.save(STATE_SAVE_LOCATION)
#         return self.corpus_state.insights[["paper_id", "download_status"]]

# class PaperAttainmentTriage:
#     """
#     Class to triage papers that failed to download (hard-to-get) and prioritize
#     manual retrieval based on semantic similarity between research questions and paper titles.
#     """

#     def __init__(
#         self,
#         corpus_state: "CorpusState",
#         client: Any,
#         embedding_model: str = "text-embedding-3-small",
#         save_location: str = os.path.join(os.getcwd(), "data", "hard_to_get_papers.csv"),
#         hard_to_get_papers: Optional[pd.DataFrame] = None
#     ) -> None:
#         """
#         Initialize PaperAttainmentTriage.

#         Args:
#             corpus_state: CorpusState object containing literature data.
#             client: OpenAI or similar embedding client.
#             embedding_model: Name of the embedding model.
#             save_location: CSV path to save the hard-to-get papers.
#             hard_to_get_papers: Optional pre-filtered DataFrame of failed downloads.
#         """
#         # Validate the corpus_state structure
#         self.corpus_state: "CorpusState" = deepcopy(
#             utils.validate_format(
#             corpus_state=corpus_state,
#             state_required_cols=[
#                 "question_id", "question_text", "search_string_id", "search_string",
#                 "paper_id", "paper_title", "paper_author", "paper_date", "doi",
#                  "download_status", "messy_question_id", "messy_paper_id"
#             ],
#             injected_value=None,
#             injected_required_cols=[]
#             )
#         )

#         self.client: Any = client
#         self.embedding_model: str = embedding_model
#         self.save_location: str = save_location

#         # Filter hard-to-get papers (failed downloads)
#         self.hard_to_get_papers: pd.DataFrame = (
#             hard_to_get_papers if hard_to_get_papers is not None 
#             else self.corpus_state.insights[self.corpus_state.insights["download_status"] == 0].copy()
#         )

#     def _generate_question_embeddings(self) -> pd.DataFrame:
#         """
#         Generate embeddings for unique research questions.

#         Returns:
#             DataFrame with columns ['question_text', 'question_embedding'].
#         """
#         questions = self.hard_to_get_papers["question_text"].drop_duplicates()
#         embeddings = []

#         for question in questions:
#             response = self.client.embeddings.create(
#                 input=question,
#                 model=self.embedding_model
#             )
#             embeddings.append(response.data[0].embedding)

#         df = pd.DataFrame({
#             "question_text": questions,
#             "question_embedding": embeddings
#         })

#         self.question_embeddings: pd.DataFrame = df
#         return df

#     def _generate_title_embeddings(self) -> pd.DataFrame:
#         """
#         Generate embeddings for all hard-to-get paper titles.

#         Returns:
#             DataFrame with columns ['paper_title', 'title_embedding'].
#         """
#         titles = self.hard_to_get_papers["paper_title"]
#         embeddings: List[Any] = []

#         for idx, title in enumerate(titles):
#             print(f"Generating embedding for title {idx + 1} of {len(titles)}")
#             response = self.client.embeddings.create(
#                 input=title,
#                 model=self.embedding_model
#             )
#             embeddings.append(response.data[0].embedding)

#         df = pd.DataFrame({
#             "paper_title": titles,
#             "title_embedding": embeddings
#         })

#         self.title_embeddings: pd.DataFrame = df
#         return df

#     def generate_embeddings(self) -> pd.DataFrame:
#         """
#         Generate embeddings for both questions and titles and merge them into one DataFrame.

#         Returns:
#             DataFrame of hard-to-get papers with question and title embeddings.
#         """
#         print("Generating question embeddings...")
#         q_df = self._generate_question_embeddings()
#         print("Generating title embeddings...")
#         t_df = self._generate_title_embeddings()

#         merged_df = self.hard_to_get_papers.merge(
#             q_df, how="left", on="question_text"
#         ).merge(
#             t_df, how="left", on="paper_title"
#         )

#         self.embeddings_df: pd.DataFrame = merged_df
#         return merged_df

#     @staticmethod
#     def calc_cosine_sim(embedding1: pd.Series, embedding2: pd.Series) -> List[float]:
#         """
#         Calculate cosine similarity between two series of embeddings.

#         Args:
#             embedding1: Series of embeddings (one per row).
#             embedding2: Series of embeddings.

#         Returns:
#             List of cosine similarity values.
#         """
#         emb1 = np.vstack(embedding1.to_numpy())
#         emb2 = np.vstack(embedding2.to_numpy())
#         dot_product = np.sum(emb1 * emb2, axis=1)
#         norms = np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
#         return (dot_product / norms).tolist()

#     @staticmethod
#     def moving_average_filter(x: Union[pd.Series, list], window: int = 5) -> List[float]:
#         """
#         Apply a moving average smoothing to a series or list.

#         Args:
#             x: Data to smooth.
#             window: Rolling window size.

#         Returns:
#             Smoothed data as a list.
#         """
#         if isinstance(x, list):
#             x = pd.Series(x)
#         return x.rolling(window=window, center=False).mean().tolist()

#     @staticmethod
#     def locate_knee(y: pd.Series) -> List[float]:
#         """
#         Locate the knee/elbow point in a descending series using KneeLocator.

#         Args:
#             y: Series of values (e.g., smoothed cosine similarities).

#         Returns:
#             List of knee_y values repeated for each item in y.
#         """
#         y_sorted = y.sort_values(ascending=False)
#         x = list(range(len(y_sorted)))

#         # Handle degenerate or empty input
#         if len(y_sorted) == 0 or y_sorted.isna().all():
#             return [np.nan for _ in y_sorted]

#         kl = KneeLocator(x=x, y=y_sorted, direction="decreasing", curve="concave")

#         # If no knee detected, replace None with np.nan
#         knee_y = kl.knee_y if kl.knee_y is not None else np.nan

#         return [knee_y for _ in y_sorted]

#     def triage_papers(
#         self,
#         low_threshold: float = 0.35,
#         medium_threshold: float = 0.5
#     ) -> pd.DataFrame:
#         """
#         Classify hard-to-get papers into 'low', 'medium', or 'high' priority for manual retrieval.

#         Args:
#             low_threshold: Cosine similarity threshold for low priority.
#             medium_threshold: Cosine similarity threshold for medium priority.

#         Returns:
#             DataFrame of hard-to-get papers with rankings and cosine similarity.
#         """
#         # Cosine similarity between question and title embeddings
#         self.hard_to_get_papers["cosine_sim"] = self.calc_cosine_sim(
#             self.embeddings_df["question_embedding"],
#             self.embeddings_df["title_embedding"]
#         )

#         # Smooth the cosine similarity
#         self.hard_to_get_papers["cosine_sim_smooth"] = self.moving_average_filter(
#             self.hard_to_get_papers["cosine_sim"]
#         )

#         # Count papers per research question
#         self.hard_to_get_papers["count"] = self.hard_to_get_papers.groupby("question_id")["paper_id"].transform("count")

#         # Compute knee/elbow for each research question
#         self.hard_to_get_papers["knee"] = self.hard_to_get_papers.groupby("question_id")["cosine_sim_smooth"].transform(self.locate_knee)

#         # Initial ranking based on low threshold
#         self.hard_to_get_papers["paper_ranking"] = np.where(
#             self.hard_to_get_papers["cosine_sim"] <= low_threshold, "low", pd.NA
#         )

#         # Count-based ranking overrides
#         self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
#             (self.hard_to_get_papers["count"] <= 10) & (self.hard_to_get_papers["cosine_sim"] > medium_threshold),
#             "high"
#         )
#         self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
#             (self.hard_to_get_papers["count"] <= 10) &
#             (self.hard_to_get_papers["cosine_sim"] > low_threshold) &
#             (self.hard_to_get_papers["cosine_sim"] <= medium_threshold),
#             "medium"
#         )

#         # Knee-based ranking
#         self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
#             self.hard_to_get_papers["paper_ranking"].isna() &
#             (self.hard_to_get_papers["cosine_sim"] > self.hard_to_get_papers["knee"]),
#             "high"
#         )

#         # Remaining papers get medium ranking
#         self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
#             self.hard_to_get_papers["paper_ranking"].isna(),
#             "medium"
#         )

#         # Merge rankings back into main corpus_state
#         self.corpus_state.insights = self.corpus_state.insights.merge(
#             self.hard_to_get_papers[["paper_id", "cosine_sim", "paper_ranking"]],
#             how="left",
#             on="paper_id"
#         )

#         # Save to CSV for manual review
#         self.corpus_state.insights.to_csv(self.save_location, index=False)
#         print(
#             f"The list of hard-to-get papers can be viewed here: {self.save_location}.\n"
#             f"Manually attain the papers that you can and save them in the relevant question folder: {os.path.join(os.getcwd(), config.CORPUS_LOCATION)}.\n"
#             f"Update this file so that download status reflects papers that you manually downloaded.\n"
#             f"Ensure manually downloaded papers follow the naming convention 'paper_id.pdf' matching this file.\n"
#             "Once you have updated the file, you should reload with .update_state() - assuming you have not moved the file from where it was saved."
#         )

#     def update_state(self):
#         # This is simply a wrapper for corpus_state.from_csv() which makes it intutive to update the corpus_state of the class after manually editing the csv
#         self.corpus_state = self.corpus_state.from_csv(filepath=self.save_location)