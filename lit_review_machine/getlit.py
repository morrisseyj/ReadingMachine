# Import custom libraries and modules
from state import QuestionState
from prompts import Prompts
import config
import utils

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
    Generate search prompts and search strings for research questions using an LLM.

    Workflow:
      1. Initialize with a list of questions and an LLM client.
      2. Build a state object (QuestionState) that tracks questions and insights.
      3. Generate structured messages for each question (`message_maker`).
      4. Query the LLM for search strings and update the state (`searchstring_maker`).
    """

    def __init__(
        self,
        questions: List[str],
        llm_client: Any,
        num_prompts: int = 10,
        search_engine: str = "Semantic Scholar",
        llm_model: str = "gpt-4.1",
        state: Optional[QuestionState] = None,
        messages: Optional[List[List[Dict[str, str]]]] = None,
    ) -> None:
        """
        Initialize the ScholarSearchString object.

        Args:
            questions (List[str]): List of research questions.
            llm_client (Any): LLM API client instance (e.g., OpenAI client).
            num_prompts (int, optional): Number of search prompts per question. Defaults to 10.
            search_engine (str, optional): Search engine context. Defaults to "Semantic Scholar".
            llm_model (str, optional): LLM model name. Defaults to "gpt-4.1".
            state (Optional[QuestionState], optional): Existing QuestionState object.
            messages (Optional[List[List[Dict[str, str]]]], optional):
                Pre-generated messages. Usually left None.
        """
        self.questions: List[str] = questions
        self.llm_client: Any = llm_client
        self.num_prompts: int = num_prompts
        self.search_engine: str = search_engine
        self.llm_model: str = llm_model

        # Create or reuse a QuestionState object
        if state:
            self.state = state
        else:
            self.state = QuestionState(insights = self._make_state())

        # Messages are generated later by `message_maker`
        self.messages: Optional[List[List[Dict[str, str]]]] = messages

    def _make_state(self) -> pd.DataFrame:
        """
        Build the initial insights dataframe with question IDs and text.

        Returns:
            pd.DataFrame: DataFrame with `question_id` and `question_text`.
        """
        state = pd.DataFrame()
        question_ids: List[str] = [f"question_{i}" for i in range(len(self.questions))]
        state["question_id"] = question_ids
        state["question_text"] = self.questions
        return state

    def message_maker(self) -> List[List[Dict[str, str]]]:
        """
        Generate LLM messages for each research question.

        Returns:
            List[List[Dict[str, str]]]: List of messages per question.
        """
        sys_prompt: str = Prompts().question_make_sys_prompt(
            search_engine=self.search_engine,
            num_prompts=self.num_prompts,
        )

        # Build user prompts from question IDs and texts
        user_prompts: List[str] = []
        for question_id, question_text in zip(
            self.state.insights["question_id"], self.state.insights["question_text"]
        ):
            user_prompt: str = f"**QUESTION**\n{question_id}: {question_text}"
            user_prompts.append(user_prompt)

        # Wrap prompts into the LLM message format
        messages: List[List[Dict[str, str]]] = []
        for prompt in user_prompts:
            message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            messages.append(message)

        # Store internally (not in state to save space/memory)
        self.messages = messages

        return self.messages

    def searchstring_maker(self) -> List[str]:
        """
        Query the LLM to generate search strings for each question,
        update the state, and persist it to pickle.

        Returns:
            List[str]: Generated search strings across all questions.
        """
        # Ensure messages exist
        if self.messages is None:
            self.message_maker()

        # Initialize empty results dataframe
        search_strings_df = pd.DataFrame(columns=["question_id", "search_string"])

        # Iterate over all questions in state
        for index, question_id in enumerate(self.state.insights["question_id"]):
            message = self.messages[index]
            print(f"Generating prompts for question {index + 1} of {self.state.insights.shape[0]}")

            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=message,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            response_data: Dict[str, List[str]] = json.loads(response.choices[0].message.content)
            # Use internal question_id instead of LLM key
            llm_prompts = list(response_data.values())[0]

            # Append results to dataframe
            search_strings_df = pd.concat(
                [
                    search_strings_df,
                    pd.DataFrame(
                        {
                            "question_id": [question_id] * len(llm_prompts),
                            "search_string": llm_prompts,
                        }
                    ),
                ],
                ignore_index=True,
            )

        # Add unique IDs for each search string
        search_strings_df["search_string_id"] = [
            f"search_string_{i}" for i in range(search_strings_df.shape[0])
        ]

        # Merge back into the insights dataframe
        self.state.insights = search_strings_df.merge(
            self.state.insights, how="left", on="question_id"
        )

        # Save updated state object
        self.state.save(os.path.join(config.STATE_SAVE_LOCATIONSTATE_SAVE_LOCATION, "01_search_strings"))

        # Return search strings for testing/debugging
        return self.state.insights["search_string"].to_list()

class AcademicLit:

    OPENALEX_BASE = "https://api.openalex.org/works"
    OPENALEX_TIMEOUT = 30

    def __init__(self, 
                 state: Optional["QuestionState"] = None, 
                 search_strings: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the AcademicLit class with a validated state or search_strings.

        Args:
            state (Optional[QuestionState]): Existing QuestionState object.
            search_strings (Optional[pd.DataFrame]): DataFrame with search strings.
        """
        # Deepcopy ensures this class has its own copy of state
        self.state = deepcopy(
            utils.validate_format(
                state=state,
                injected_value=search_strings,
                state_required_cols=["question_id", "question_text", "search_string", "search_string_id"],
                injected_required_cols=["question_id", "question_text", "search_string", "search_string_id"]
            )
        )
    
    # CLASS UTILS ------------
    def _merge_search_results_with_state(self, search_results: pd.DataFrame) -> pd.DataFrame:
        """
        Merge search results with the existing state.insights DataFrame.

        - Adds question metadata (question_id, question_text, search_string_id, search_string).
        - Deduplicates across engines.
        - Returns the merged DataFrame (not written to disk).
        """
        if "search_engine" in getattr(self.state, "insights", pd.DataFrame()).columns:
            # Merge to bring in question info and concatenate with existing results
            enriched = search_results.merge(
                self.state.insights[["question_id", "question_text", "search_string_id", "search_string"]],
                on="search_string",
                how="left"
            )
            merged = pd.concat([self.state.insights, enriched], ignore_index=True)
        else:
            # First engine call — merge just to add question info
            merged = search_results.merge(
                self.state.insights,
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
        Search Crossref for each search string in state and update the state.
        Returns self.state.insights (DataFrame).
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

        for search_string in self.state.insights["search_string"]:
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

        # Merge results into state
        self.state.insights = self._merge_search_results_with_state(output_df)
        return self.state.insights
    
    @staticmethod
    def _openalex_authors(authorships) -> List[str]:
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
        Search OpenAlex for each search string in state and merge with existing state.
        Each query returns up to 200 results (one page).
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

        for search_string in self.state.insights["search_string"]:
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

        self.state.insights = merged
        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "02_academic_lit"))
        return self.state.insights
    
# class DOI:
#     """
#     Retrieve DOIs and open-access download links for papers stored in a QuestionState
#     object or provided as a DataFrame.
#     """

#     def __init__(
#         self, 
#         state: Optional["QuestionState"] = None, 
#         papers: Optional[pd.DataFrame] = None
#     ) -> None:
#         """
#         Initialize DOI retriever.

#         Args:
#             state: A pre-existing QuestionState object containing paper metadata.
#             papers: A DataFrame containing paper metadata (used if state is None).
#         """
#         # Validate and set up state
#         self.state = deepcopy(
#             validate_format(
#             state=state,
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
#         if self.state.insights.empty:
#             return []

#         df = self.state.insights[["paper_title", "paper_author", "paper_date"]].copy()
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
#         Retrieves DOIs for all papers in the current state using OpenAlex.

#         Returns:
#             List[Optional[str]]: A list of DOIs corresponding to papers.
#         """
#         dois = self.state.insights["doi"]

#         if not self.search_string:
#             print("No papers available to retrieve DOIs.")
#             self.state.insights["doi"] = []
#             return []

#         for idx, (string, doi) in enumerate(zip(self.search_string, dois), start=1):
#             print(f"Retrieving DOI {idx} of {len(self.search_string)}")
#             if not pd.isna(doi):
#                 continue  # Skip if DOI already exists from the AcademicLit search
#             else:
#                 doi_result = self.call_alex(string)
#                 dois[idx - 1] = doi_result

#         self.state.insights["doi"] = dois

#         return dois

#     def get_download_link(self) -> List[Optional[str]]:
#         """
#         Retrieves open-access PDF download links for each paper via Unpywall.

#         Returns:
#             List[Optional[str]]: A list of open-access PDF download links (or None if unavailable).
#         """
#         # Ensure DOI column exists
#         self.state = validate_format(
#             state=self.state,
#             injected_value=self.state.insights,
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

#         for idx, doi in enumerate(self.state.insights.get("doi", []), start=1):
#             print(f"Retrieving downlod link for paper {idx} of {self.state.insights.shape[0]}")

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

#         self.state.insights["download_link"] = download_links
#         self.state.save(STATE_SAVE_LOCATION)

#         return download_links

class GreyLiterature:
    """
    A class for retrieving and managing grey literature using an LLM and live web search.

    Grey literature is defined here as reports, policy briefs, working papers, and case
    studies published by think tanks, INGOs, multilateral organizations, and other 
    research institutions.

    The class integrates with a QuestionState object that tracks research questions
    and insights, ensuring that retrieved grey literature is associated with the 
    correct research question IDs.
    """

    # Default path for caching grey literature results
    def __init__(
        self,
        llm_client: Any,  # Client interface for interacting with the LLM API
        state: Optional["QuestionState"] = None,  # Current research state (can be injected)
        questions: Optional[List[str]] = None,    # User-defined research questions
        ai_reasoning_model: str = "o3-deep-research",       # LLM model to use
        ai_chat_completion_model: str = "gpt-4o",  # Chat completion model for JSON cleaning
        grey_lit_pickle_folder: str = os.path.join(os.getcwd(), "data", "pickles") # The pickle location for the valid processed json response from the LLM
        
        #GREY_LIT_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit.pkl"), # The pickle location for the valid processed json response from the LLM
        #GREY_LIT_RAW_PICKLE_FILE: str = os.path.join(os.getcwd(), "data", "pickles", "grey_lit_raw.pkl") # If the LLM fails to return a valid json the raw output gets saved here - as this is an expensive call
    ) -> None:
        """
        Initialize the GreyLiterature object.

        Args:
            llm_client (Any): Client for interacting with the language model.
            state (Optional[QuestionState]): QuestionState object holding research state.
            questions (Optional[List[str]]): User-defined research questions.
            ai_model (str): Name of the LLM model to use.
        """

        # If questions are provided directly, format them into a DataFrame
        if questions:
            question_id: List[str] = [f"Question_{i}" for i in range(len(questions))]
            questions = pd.DataFrame({
                "question_id": question_id,
                "question_text": questions
            })

        # Validate the state and inject the questions if provided
        self.state: "QuestionState" = deepcopy(
            utils.validate_format(
                state=state,
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
        Retrieve grey literature relevant to the research questions using the LLM.

        Steps:
        1. Build a prompt from research questions.
        2. Call the LLM with web search capability.
        3. Parse the JSON output from the LLM.
        4. Merge results with existing QuestionState using `question_id`.
        5. Save updated state to disk.

        Returns:
            Optional[pd.DataFrame]: Subset of state with grey literature results
            (where `paper_id` starts with "grey_lit_"), or None if parsing fails.
        """

        if os.path.exists(os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl")):
            recover = None
            while recover not in ["r", "n"]:
                recover = input("AI generated grey literature already exists. Would you like to recover (r) or generate new (n)? (r/n): ").lower()
            if recover == "r":
                with open(os.path.join(self.grey_lit_pickle_folder, "grey_lit.pkl"), "rb") as f:
                    self.grey_lit = pickle.load(f)

                self.state.insights = pd.concat([self.state.insights, self.grey_lit], ignore_index=True)
                return self.grey_lit

        # Build question strings: "question_id: question_text"
        question_strings = (
            self.state.insights[["question_id", "question_text"]]
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
            self.state.insights[["question_id", "question_text"]].drop_duplicates(),
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

        # Update state
        self.state.insights = pd.concat([self.state.insights, self.grey_lit], ignore_index=True)
        self.state.insights["paper_date"] = pd.to_numeric(self.state.insights["paper_date"], errors="coerce").astype("Int64")
        self.state.insights.replace(["", "NA", pd.NA, np.nan, "null"], None, inplace=True)
        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "03_grey_lit"))

        # Return grey literature
        return self.grey_lit

class Literature:
    """
    A class to manage literature (including grey literature) for research questions,
    detect exact and fuzzy duplicates, and export files for manual checking.

    Workflow:
    1. Split literature by question_id.
    2. Generate a string for duplicate detection.
    3. Drop exact duplicates.
    4. Detect fuzzy duplicates using pairwise string similarity.
    5. Export potential matches for manual verification.
    6. Update QuestionState with cleaned results.
    """

    FUZZY_CHECK_PATH: str = os.path.join(os.getcwd(), "data", "fuzzy_check")
    os.makedirs(FUZZY_CHECK_PATH, exist_ok=True)

    def __init__(self, state: "QuestionState", literature: Optional[pd.DataFrame] = None) -> None:
        
        self.state: "QuestionState" = deepcopy(
            utils.validate_format(
            state=state,
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
        Split the literature by question_id and generate a string for duplicate detection.
        Only question_id is needed; question_text is not included here.
        """
        dfs: List[pd.DataFrame] = [
            self.state.insights[self.state.insights["question_id"] == qid].copy()
            for qid in self.state.insights["question_id"].drop_duplicates()
        ]

        for df in dfs:
            authors_str = df["paper_author"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            title_str = df["paper_title"].astype(str)
            date_str = df["paper_date"].astype(str)

            # Concatenate for duplicate checking
            df["duplicate_check_string"] = authors_str + " " + title_str + " " + date_str

        return dfs

    def drop_exact_duplicates(self) -> List[pd.DataFrame]:
        for df in self.question_dfs:
            df.drop_duplicates(subset="duplicate_check_string", keep="first", inplace=True)
        return self.question_dfs

    def _get_fuzzy_match(self, similarity_threshold: int = 90) -> List[List[Tuple[str, str]]]:
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
        
        self.state.insights = pd.concat(dfs, ignore_index=True)

        if "duplicate_check_string" in self.state.insights.columns:
            self.state.insights.drop(columns="duplicate_check_string", inplace=True)

        if "sim_group" in self.state.insights.columns:
            self.state.insights.drop(columns="sim_group", inplace=True)

        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "04_literature_deduped"))
        return self.state.insights

class AiLiteratureCheck:
    """
    Class to check the completeness of literature for a set of research questions
    using an LLM. Takes a QuestionState object and optionally a DataFrame of papers
    and outputs missing literature suggested by the AI.
    """

    def __init__(
        self,
        llm_client: Any,
        ai_reasoning_model: str = "o3-deep-research",
        ai_chat_completion_model: str = "gpt-4o",
        state: Optional["QuestionState"] = None,
        papers: Optional[pd.DataFrame] = None, 
    ) -> None:
        """
        Initialize the AI Literature Check.

        Args:
            llm_client: An instance of the language model client (e.g., OpenAI client).
            ai_model: The name of the LLM model to use.
            state: QuestionState object containing current literature data.
            papers: Optional DataFrame with literature to inject if state is None.
        """
        self.llm_client: Any = llm_client
        self.ai_reasoning_model: str = ai_reasoning_model
        self.ai_chat_completion_model: str = ai_chat_completion_model

        # Validate that the state or injected papers contain all required columns
        self.state: "QuestionState" = deepcopy(
            utils.validate_format(
            state=state,
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
        Converts the literature DataFrame into JSON suitable for LLM prompt insertion.
        Groups papers by question_id and question_text, flattening into a list of paper dicts.

        Returns:
            str: JSON string of grouped literature for input to the LLM.
        """
        df = self.state.insights[[
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
        Uses the LLM to identify missing literature for each research question.
        Parses the LLM JSON output, flattens it, merges with the state, and
        assigns unique AI literature paper IDs.

        Returns:
            Optional[pd.DataFrame]: DataFrame of AI-suggested missing papers with columns:
            ["paper_id", "paper_title", "paper_author", "paper_date"].
            Returns None if LLM call fails or invalid output is returned.
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
        
        # if the response is empty no new papers were found so we print a message and save the current state and return the state.insights
        if len(clean_response["results"]) == 0:
            print("No missing papers returned by the LLM.")
            self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "05_ai_lit_check"))
            return self.state.insights

        # Otherwise we convert to a df, clean and concat with state.insights
        ai_lit = pd.DataFrame(clean_response["results"])
        # Clean:
        # Get canonical question text and join
        canonical_questions = self.state.insights[["question_id", "question_text"]].drop_duplicates()
        ai_lit = ai_lit.merge(
            canonical_questions,
            how="left",
            on="question_id"
        )
        # Assign unique AI paper IDs
        ai_lit["paper_id"] = [f"ai_lit_{i}" for i in range(ai_lit.shape[0])]
        
        # Create ai_lit attribute for inspection
        self.ai_lit = ai_lit

        # Append AI literature to state
        updated_state = pd.concat([self.state.insights, ai_lit], ignore_index=True)
        # CLean up any dates to numeric to not break parquet
        updated_state["paper_date"] = pd.to_numeric(updated_state["paper_date"], errors="coerce").astype("Int64") 
        # Clean up any empty strings to None to not break parquet
        updated_state.replace(["", "NA", pd.NA, np.nan, "null", "No author found"], None, inplace=True)
        # Update the oder for pretty export
        updated_state = updated_state[["question_id", "question_text", "search_string_id", "search_string", "search_engine", "paper_id", "paper_title", "paper_author", "paper_date", "doi"]]

        # Asign to state attribute
        self.state.insights = updated_state

        # Save
        self.state.save(os.path.join(config.STATE_SAVE_LOCATION, "05_ai_lit_check"))

        # Return only the new AI-suggested papers
        return self.ai_lit

class DownloadManager:
    """
    Class to manage downloading of papers listed in a QuestionState object.
    Downloads are organized by sanitized question_id and paper_id to ensure
    filesystem-safe filenames, while maintaining traceability to original IDs.
    """

    def __init__(
        self,
        state: "QuestionState" = None,
        papers: Optional[pd.DataFrame] = None,
        DOWNLOAD_LOCATION: str = os.path.join(os.getcwd(), "data", "docs")
    ) -> None:
        """
        Initialize the Downloader.

        Args:
            state: QuestionState object containing literature data.
            papers: Optional DataFrame of literature to inject.
            DOWNLOAD_LOCATION: Base directory to save downloaded files.
        """
        # Validate that the state or injected papers contain all required columns
        self.state: "QuestionState" = deepcopy(
            utils.validate_format(
            state=state,
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

        # Check if download_status variable is in the passed state if its not create with 0 values, assuming no downloads have happened yet, if it does exsist, simply use it. 
        if "download_status" not in self.state.insights.columns:
            self.state.insights["download_status"] = 0
        else: 
            pass

        # Ensure the base download folder exists
        self.DOWNLOAD_LOCATION: str = DOWNLOAD_LOCATION
        os.makedirs(self.DOWNLOAD_LOCATION, exist_ok=True)

        # Create subfolders for each question_id
        self._create_download_folder()

        # Preserve original IDs and sanitize for filesystem-safe filenames
        self.state.insights["messy_question_id"] = self.state.insights["question_id"]
        self.state.insights["messy_paper_id"] = self.state.insights["paper_id"]
        self.state.insights["question_id"] = self.state.insights["question_id"].apply(self._sanitize_filename)
        self.state.insights["paper_id"] = self.state.insights["paper_id"].apply(self._sanitize_filename)

        # write the insights to csv
        self.state.write_to_csv(save_location= self.DOWNLOAD_LOCATION, 
                                write_full_text=False, write_chunks=False)
        
        print(
            f"Architecture for downloading papers has been created at {self.DOWNLOAD_LOCATION}.\n"
            f"You sould manually download files and update thier status in the file at {os.path.join(self.DOWNLOAD_LOCATION, "insights.csv")}. "
            "Assuming you do not change the files location the easiest way to do this is to call DownloadManager.update()\n"
            f"Note when saving these files you MUST SAVE THEM IN THE FOLDER CORRESPONDING TO THIER QUESTION ID. You should also ensure the filenames match the paper_id in the form paper_id.[relevant extension]. " 
            "Matching filenames with paper_ids is not neccesary but will allow you to track papers back to search prompts. You can add papers to these folders that are not in your "
            )

    def _create_download_folder(self) -> None:
        """
        Create subfolders for each sanitized question_id to organize downloaded files.
        """
        for qid in self.state.insights["question_id"].unique():
            os.makedirs(os.path.join(self.DOWNLOAD_LOCATION, qid), exist_ok=True)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize a string to be a valid filename by removing illegal filesystem characters.

        Args:
            filename: Original filename string.

        Returns:
            Sanitized filename string.
        """
        sanitized = re.sub(r'[\\/:*?"<>|]', "_", filename)
        return sanitized.strip()
    
    def update(self):
        # This convenience function just calls the from csv method of the questionstate
        self.state = self.state.from_csv(filepath=os.path.join(self.DOWNLOAD_LOCATION))
        # And updates the state object on file by saving
        self.state.save(config.STATE_SAVE_LOCATION)
        return(self.state.insights)

# THIS WAS ALL TOO FRAGILE TO MAKE WORK SO I BAILED ON IT AND RESORTED TO MANUAL DOWNLOADING
#     def download_files(self) -> pd.DataFrame:
#         """
#         Attempt to download all files in the state DataFrame. Tracks download status
#         and local filenames. Updates state and writes a CSV with download results.

#         Returns:
#             DataFrame containing columns ['paper_id', 'download_status'] with updated statuses.
#         """
#         # Ensure subfolders exist
#         self._create_download_folder()

#         # Initialize download tracking columns
#         if "download_status" not in self.state.insights.columns:
#             self.state.insights["download_status"] = 0
#         if "filename" not in self.state.insights.columns:
#             self.state.insights["filename"] = np.nan

#         # Iterate through each row and attempt download
#         for idx, row in self.state.insights.iterrows():
#             url: str = row["download_link"]
#             status: int = row["download_status"]
#             qid: str = row["question_id"]
#             pid: str = row["paper_id"]

#             print(f"Downloading file {idx + 1} of {self.state.insights.shape[0]}")

#             if status == 0:
#                 if pd.notna(url) and url != "NA":
#                     try:
#                         response = requests.get(url, stream=True, timeout=10)
#                         response.raise_for_status()

#                         file_path = os.path.join(self.DOWNLOAD_LOCATION, qid, f"{pid}.pdf")
#                         with open(file_path, "wb") as f:
#                             for chunk in response.iter_content(chunk_size=8192):
#                                 f.write(chunk)

#                         self.state.insights.at[idx, "filename"] = file_path
#                         self.state.insights.at[idx, "download_status"] = 1
#                     except Exception as e:
#                         print(f"Failed to download {url}: {e}")
#                         self.state.insights.at[idx, "filename"] = np.nan
#                         self.state.insights.at[idx, "download_status"] = 0
#                 else:
#                     self.state.insights.at[idx, "filename"] = np.nan
#                     self.state.insights.at[idx, "download_status"] = 0
#             else:
#                 self.state.insights.at[idx, "download_status"] = 1

#         # Save download status CSV for inspection
#         download_status_csv = os.path.join(self.DOWNLOAD_LOCATION, "download_status.csv")
#         self.state.insights.to_csv(download_status_csv, index=False)

#         print(
#             f"Attempted downloads complete. Inspect the results here: {download_status_csv}.\n"
#             "For files that failed to download, open this CSV, update the 'download_link' as needed, and save it.\n"
#             "Then reload the updated CSV into a QuestionState using:\n"
#             "    state = QuestionState.load_from_csv('path/to/download_status.csv')\n"
#             "After that, pass the new state to the Downloader and retry downloads:\n"
#             "    downloader = Downloader(state=state)\n"
#             "Filenames correspond to sanitized question_id and paper_id, preserving traceability."
#         )
#         # Save the state
#         self.state.save(STATE_SAVE_LOCATION)
#         return self.state.insights[["paper_id", "download_status"]]

class PaperAttainmentTriage:
    """
    Class to triage papers that failed to download (hard-to-get) and prioritize
    manual retrieval based on semantic similarity between research questions and paper titles.
    """

    def __init__(
        self,
        state: "QuestionState",
        client: Any,
        embedding_model: str = "text-embedding-3-small",
        save_location: str = os.path.join(os.getcwd(), "data", "hard_to_get_papers.csv"),
        hard_to_get_papers: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize PaperAttainmentTriage.

        Args:
            state: QuestionState object containing literature data.
            client: OpenAI or similar embedding client.
            embedding_model: Name of the embedding model.
            save_location: CSV path to save the hard-to-get papers.
            hard_to_get_papers: Optional pre-filtered DataFrame of failed downloads.
        """
        # Validate the state structure
        self.state: "QuestionState" = deepcopy(
            utils.validate_format(
            state=state,
            state_required_cols=[
                "question_id", "question_text", "search_string_id", "search_string",
                "paper_id", "paper_title", "paper_author", "paper_date", "doi",
                 "download_status", "messy_question_id", "messy_paper_id"
            ],
            injected_value=None,
            injected_required_cols=[]
            )
        )

        self.client: Any = client
        self.embedding_model: str = embedding_model
        self.save_location: str = save_location

        # Filter hard-to-get papers (failed downloads)
        self.hard_to_get_papers: pd.DataFrame = (
            hard_to_get_papers if hard_to_get_papers is not None 
            else self.state.insights[self.state.insights["download_status"] == 0].copy()
        )

    def _generate_question_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings for unique research questions.

        Returns:
            DataFrame with columns ['question_text', 'question_embedding'].
        """
        questions = self.hard_to_get_papers["question_text"].drop_duplicates()
        embeddings = []

        for question in questions:
            response = self.client.embeddings.create(
                input=question,
                model=self.embedding_model
            )
            embeddings.append(response.data[0].embedding)

        df = pd.DataFrame({
            "question_text": questions,
            "question_embedding": embeddings
        })

        self.question_embeddings: pd.DataFrame = df
        return df

    def _generate_title_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings for all hard-to-get paper titles.

        Returns:
            DataFrame with columns ['paper_title', 'title_embedding'].
        """
        titles = self.hard_to_get_papers["paper_title"]
        embeddings: List[Any] = []

        for idx, title in enumerate(titles):
            print(f"Generating embedding for title {idx + 1} of {len(titles)}")
            response = self.client.embeddings.create(
                input=title,
                model=self.embedding_model
            )
            embeddings.append(response.data[0].embedding)

        df = pd.DataFrame({
            "paper_title": titles,
            "title_embedding": embeddings
        })

        self.title_embeddings: pd.DataFrame = df
        return df

    def generate_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings for both questions and titles and merge them into one DataFrame.

        Returns:
            DataFrame of hard-to-get papers with question and title embeddings.
        """
        print("Generating question embeddings...")
        q_df = self._generate_question_embeddings()
        print("Generating title embeddings...")
        t_df = self._generate_title_embeddings()

        merged_df = self.hard_to_get_papers.merge(
            q_df, how="left", on="question_text"
        ).merge(
            t_df, how="left", on="paper_title"
        )

        self.embeddings_df: pd.DataFrame = merged_df
        return merged_df

    @staticmethod
    def calc_cosine_sim(embedding1: pd.Series, embedding2: pd.Series) -> List[float]:
        """
        Calculate cosine similarity between two series of embeddings.

        Args:
            embedding1: Series of embeddings (one per row).
            embedding2: Series of embeddings.

        Returns:
            List of cosine similarity values.
        """
        emb1 = np.vstack(embedding1.to_numpy())
        emb2 = np.vstack(embedding2.to_numpy())
        dot_product = np.sum(emb1 * emb2, axis=1)
        norms = np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        return (dot_product / norms).tolist()

    @staticmethod
    def moving_average_filter(x: Union[pd.Series, list], window: int = 5) -> List[float]:
        """
        Apply a moving average smoothing to a series or list.

        Args:
            x: Data to smooth.
            window: Rolling window size.

        Returns:
            Smoothed data as a list.
        """
        if isinstance(x, list):
            x = pd.Series(x)
        return x.rolling(window=window, center=False).mean().tolist()

    @staticmethod
    def locate_knee(y: pd.Series) -> List[float]:
        """
        Locate the knee/elbow point in a descending series using KneeLocator.

        Args:
            y: Series of values (e.g., smoothed cosine similarities).

        Returns:
            List of knee_y values repeated for each item in y.
        """
        y_sorted = y.sort_values(ascending=False)
        x = list(range(len(y_sorted)))

        # Handle degenerate or empty input
        if len(y_sorted) == 0 or y_sorted.isna().all():
            return [np.nan for _ in y_sorted]

        kl = KneeLocator(x=x, y=y_sorted, direction="decreasing", curve="concave")

        # If no knee detected, replace None with np.nan
        knee_y = kl.knee_y if kl.knee_y is not None else np.nan

        return [knee_y for _ in y_sorted]

    def triage_papers(
        self,
        low_threshold: float = 0.35,
        medium_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Classify hard-to-get papers into 'low', 'medium', or 'high' priority for manual retrieval.

        Args:
            low_threshold: Cosine similarity threshold for low priority.
            medium_threshold: Cosine similarity threshold for medium priority.

        Returns:
            DataFrame of hard-to-get papers with rankings and cosine similarity.
        """
        # Cosine similarity between question and title embeddings
        self.hard_to_get_papers["cosine_sim"] = self.calc_cosine_sim(
            self.embeddings_df["question_embedding"],
            self.embeddings_df["title_embedding"]
        )

        # Smooth the cosine similarity
        self.hard_to_get_papers["cosine_sim_smooth"] = self.moving_average_filter(
            self.hard_to_get_papers["cosine_sim"]
        )

        # Count papers per research question
        self.hard_to_get_papers["count"] = self.hard_to_get_papers.groupby("question_id")["paper_id"].transform("count")

        # Compute knee/elbow for each research question
        self.hard_to_get_papers["knee"] = self.hard_to_get_papers.groupby("question_id")["cosine_sim_smooth"].transform(self.locate_knee)

        # Initial ranking based on low threshold
        self.hard_to_get_papers["paper_ranking"] = np.where(
            self.hard_to_get_papers["cosine_sim"] <= low_threshold, "low", pd.NA
        )

        # Count-based ranking overrides
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            (self.hard_to_get_papers["count"] <= 10) & (self.hard_to_get_papers["cosine_sim"] > medium_threshold),
            "high"
        )
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            (self.hard_to_get_papers["count"] <= 10) &
            (self.hard_to_get_papers["cosine_sim"] > low_threshold) &
            (self.hard_to_get_papers["cosine_sim"] <= medium_threshold),
            "medium"
        )

        # Knee-based ranking
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            self.hard_to_get_papers["paper_ranking"].isna() &
            (self.hard_to_get_papers["cosine_sim"] > self.hard_to_get_papers["knee"]),
            "high"
        )

        # Remaining papers get medium ranking
        self.hard_to_get_papers["paper_ranking"] = self.hard_to_get_papers["paper_ranking"].mask(
            self.hard_to_get_papers["paper_ranking"].isna(),
            "medium"
        )

        # Merge rankings back into main state
        self.state.insights = self.state.insights.merge(
            self.hard_to_get_papers[["paper_id", "cosine_sim", "paper_ranking"]],
            how="left",
            on="paper_id"
        )

        # Save to CSV for manual review
        self.state.insights.to_csv(self.save_location, index=False)
        print(
            f"The list of hard-to-get papers can be viewed here: {self.save_location}.\n"
            f"Manually attain the papers that you can and save them in the relevant question folder: {os.path.join(os.getcwd(), 'data', 'docs')}.\n"
            f"Update this file so that download status reflects papers that you manually downloaded.\n"
            f"Ensure manually downloaded papers follow the naming convention 'paper_id.pdf' matching this file.\n"
            "Once you have updated the file, you should reload with .update_state() - assuming you have not moved the file from where it was saved."
        )

    def update_state(self):
        # This is simply a wrapper for state.from_csv() which makes it intutive to update the state of the class after manually editing the csv
        self.state = self.state.from_csv(filepath=self.save_location)