"""
Shared utility functions for the ReadingMachine pipeline.

This module provides reusable helpers that support corpus acquisition,
corpus reading, thematic synthesis, persistence, and pipeline
orchestration. The functions are used across the getlit workflow, the
core ReadingMachine pipeline, and downstream synthesis components.

Major utility categories include:

    - state validation and normalization
    - LLM API wrappers and reasoning-model orchestration
    - pipeline recovery and resume helpers
    - corpus deduplication and similarity matching
    - text normalization and hashing
    - schema-safe DataFrame operations
    - persistence and serialization helpers
    - context-window management utilities

The utilities centralize infrastructure and workflow concerns so that
pipeline components can focus on the methodological stages of
ReadingMachine rather than implementation details.

Design principles
-----------------

The utilities are designed to support several methodological and
engineering goals:

Consistency
    Shared validation, normalization, and schema-management helpers
    ensure that pipeline stages operate on predictable inputs.

Reproducibility
    Deterministic normalization, hashing, fingerprinting, and sampling
    utilities help maintain stable analytical workflows across runs.

Resumability
    State persistence, safe serialization, and recovery helpers support
    long-running workflows that may span multiple sessions.

Inspectability
    Utilities favor explicit intermediate representations and state
    management rather than opaque transformations.

Scalability
    Helpers such as sampling, deduplication, and asynchronous model
    orchestration support operation over large corpora while respecting
    practical resource constraints.

Relationship to ReadingMachine
------------------------------
These utilities do not implement the core methodological stages of
ReadingMachine directly. Instead, they provide the supporting operations
required to execute the workflow reliably, including corpus validation,
duplicate detection, LLM interaction, persistence, and recovery.

Together, these functions form the operational layer that enables the
ReadingMachine reading and synthesis pipelines to run as structured,
resumable, and inspectable analytical processes.
"""

from .state import CorpusState
from . import config

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import json
import pyarrow as pa
import os
import time
from openai import OpenAI, APITimeoutError, APIConnectionError
import networkx as nx
from rapidfuzz import process, fuzz
import pickle
import os
import hashlib
import re
import uuid
import random
from itertools import combinations

def validate_format(
    corpus_state: Optional["CorpusState"] = None, 
    questions: Optional[pd.DataFrame] = None,
    injected_value: Optional[pd.DataFrame] = None,
    state_required_cols: List[str] = [],
    injected_required_cols: List[str] = []
) -> "CorpusState":
    """
    Validate input state and return a CorpusState.

    Accepts either an existing CorpusState or the components needed to
    construct a new one from an injected DataFrame. This helper is used by
    pipeline steps that can operate on either a pre-existing state object or
    a manually supplied table.

    Parameters
    ----------
    corpus_state : CorpusState, optional
        Existing CorpusState to validate. If provided, `questions` and
        `injected_value` must not also be provided.

    questions : pd.DataFrame, optional
        Questions table used when constructing a new CorpusState from
        `injected_value`. Must be provided together with `injected_value`.

    injected_value : pd.DataFrame, optional
        DataFrame to use as the `insights` table when constructing a new
        CorpusState. Must be provided together with `questions`.

    state_required_cols : list[str], default=[]
        Columns required in `corpus_state.insights` when validating an
        existing state. When constructing a new state, any missing columns
        from this list are added to `injected_value` and filled with `NaN`.

    injected_required_cols : list[str], default=[]
        Columns that must already exist in `injected_value` before a new
        CorpusState can be constructed.

    Returns
    -------
    CorpusState
        The validated existing CorpusState, or a newly constructed
        CorpusState using `questions` and `injected_value`.

    Raises
    ------
    ValueError
        If both initialization paths are supplied, neither valid path is
        supplied, required columns are missing, or an existing state's
        `insights.paper_id` column contains missing values.

    Notes
    -----
    For an existing CorpusState, this function validates only
    `corpus_state.insights` and does not inspect `questions`, `full_text`, or
    `chunks`.

    For injected DataFrames, the function mutates `injected_value` in place
    by adding any missing `state_required_cols` as `NaN` columns before
    constructing the CorpusState.
    """
    
    # --- PATH A: Existing State Provided ---
    if corpus_state is not None:
        # Strict check: Ensure they didn't try to provide Path B arguments too
        if questions is not None or injected_value is not None:
            raise ValueError("Provide EITHER 'corpus_state' OR ('questions' AND 'injected_value'), not both.")

        # Column Validation
        if not set(state_required_cols).issubset(corpus_state.insights.columns):
            raise ValueError(f"corpus_state.insights missing required columns: {state_required_cols}")
            
        if "paper_id" in corpus_state.insights.columns and corpus_state.insights["paper_id"].isna().any():
            raise ValueError("corpus_state.insights contains NA values in 'paper_id'.")
            
        return corpus_state

    # --- PATH B: New State via Injection ---
    elif questions is not None and injected_value is not None:
        if not set(injected_required_cols).issubset(injected_value.columns):
            raise ValueError(f"Injected DataFrame missing: {injected_required_cols}")

        # Fill missing columns
        for field in state_required_cols:
            if field not in injected_value.columns: # Use .columns check directly
                injected_value[field] = np.nan

        return CorpusState(questions=questions, insights=injected_value)

    # --- PATH C: Failure (Nothing provided or partial Path B) ---
    else:
        raise ValueError(
            "Invalid arguments. You must provide a 'corpus_state' object "
            "OR both 'questions' and 'injected_value'."
        )
    
def call_chat_completion(
    llm_client,
    ai_model: str,
    sys_prompt: str,
    user_prompt: str,
    fall_back: Dict[str, Any],
    return_json: bool,
    json_schema: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    return_with_error: bool = False
    ) -> Union[
    str,
    Dict[str, Any],
    Tuple[Union[str, Dict[str, Any]], Optional[Dict[str, str]]]
    ]:
    """
    Call a chat-completion model with standardized prompt, JSON, and error handling.

    Builds a two-message chat-completion request from a system prompt and a
    user prompt, sends it to the supplied LLM client, and returns either raw
    text or parsed JSON. The helper optionally requests JSON-mode or
    schema-constrained JSON responses and can return structured error
    metadata alongside the result.

    Parameters
    ----------
    llm_client : Any
        Client object exposing `chat.completions.create()`.

    ai_model : str
        Model identifier passed to the chat-completion API.

    sys_prompt : str
        System prompt used as the first message.

    user_prompt : str
        User prompt used as the second message.

    fall_back : dict[str, Any]
        Value returned when `return_json=True` and the API call fails or the
        response cannot be parsed as JSON.

    return_json : bool
        If True, request a JSON response and parse the returned content with
        `json.loads()`. If False, return the response content as text.

    json_schema : dict, optional
        Optional schema passed as `response_format={"type": "json_schema",
        "json_schema": json_schema}` when `return_json=True`. If omitted and
        `return_json=True`, JSON object mode is requested instead.

    max_tokens : int, optional
        Maximum number of output tokens to request. If omitted, no
        `max_tokens` argument is passed.

    return_with_error : bool, default=False
        If True, return `(result, error)`. If False, return only `result`.

    Returns
    -------
    str or dict[str, Any] or tuple[str | dict[str, Any], dict[str, str] | None]
        If `return_with_error=False`, returns either response text
        (`return_json=False`) or parsed JSON / fallback (`return_json=True`).

        If `return_with_error=True`, returns a tuple containing the result
        and an error dictionary, or `None` if no error occurred.

    Raises
    ------
    None
        API and JSON parsing failures are caught internally. Failures are
        represented through fallback return values and, when requested,
        structured error metadata.

    Notes
    -----
    The request is sent with `temperature=0`.

    If `return_json=True`, JSON parsing failures return `fall_back` and
    record a `"parse_error"` when `return_with_error=True`.

    If the API call fails or returns empty content, the function returns
    `fall_back` for JSON mode and an empty string for text mode. In this
    case the recorded error type is `"api_error"`.

    This wrapper does not validate returned JSON against `json_schema`
    locally; it only passes the schema to the model API through
    `response_format`.
    """
    # create the messages
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Allocate the variabls for the API call to kwargs for cleaner handling of optional parameters
    kwargs = {
        "model": ai_model,
        "messages": messages,
        "temperature": 0
    }

    # Conditionally add response format to kwargs based on return_json and json_schema
    if return_json:
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
    
    # Conditionally add max_tokens to kwargs if provided
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # Set both values to none and compute later
    result: Union[str, Dict[str, Any], None] = None
    error: Optional[Dict[str, str]] = None

    # Pass api the kwargs
    try:
        response = llm_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        # Check if call worked:
        # LLM returned something
        if content is None:
            raise ValueError("Empty response content")

        # LLM returned valid JSON (if expected)
        if return_json:
            try:
                result = json.loads(content.strip())
            except Exception as e:
                result = fall_back
                error = {
                    "type": "parse_error",
                    "message": f"Failed to parse JSON: {e}"
                }
        else:
            result = content

    # Api call failed and error was returned
    except Exception as e:
        result = fall_back if return_json else ""
        error = {
            "type": "api_error",
            "message": str(e)
        }

    # Conditionally retrn output or output with error based on return_with_error flag
    if return_with_error:
        return result, error

    return result

def call_reasoning_model(
    prompt: str,
    llm_client: OpenAI,
    ai_model: str = "o3-deep-research",
    id_timeout: float = 30,
    max_retry: int = 2,
    poll_interval: int = 60,
    max_poll_errors: int = 20,
):
    """
    Run a background reasoning-model job and poll until completion.

    Creates a background response using the OpenAI Responses API, then
    periodically retrieves the response status until the job completes,
    fails, is cancelled, expires, or exceeds the allowed number of polling
    errors.

    Parameters
    ----------
    prompt : str
        Prompt submitted to the reasoning model.

    llm_client : OpenAI
        OpenAI client instance exposing `responses.create()` and
        `responses.retrieve()`.

    ai_model : str, default="o3-deep-research"
        Model identifier used when creating the response.

    id_timeout : float, default=30
        Timeout, in seconds, used when creating the background response job.

    max_retry : int, default=2
        Maximum number of attempts to create the background response job when
        transient timeout or connection errors occur.

    poll_interval : int, default=60
        Number of seconds to wait between response-status polling attempts.

    max_poll_errors : int, default=20
        Maximum number of consecutive polling timeout or connection errors
        allowed before the function returns failure.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - `status`: `"success"` or `"failed"`
        - `response`: output text on success, the full response object in
        some failure cases, or None if no response is available
        - `error`: None on success, otherwise error details
        - `response_id`: created response ID, or None if job creation failed

    Notes
    -----
    The response is created with `background=True` and with the
    `web_search_preview` tool enabled.

    Only `APITimeoutError` and `APIConnectionError` are retried during job
    creation and polling. Other exceptions are not caught.

    If the response status is `"completed"` but `output_text` is empty, the
    function returns `status="failed"` and includes the full response object
    for inspection.

    This function blocks while polling. It does not return until the
    background response reaches a terminal status or polling fails too many
    times.
    """

    def get_resp_id():
        attempt = 1
        last_err = None

        while attempt <= max_retry:
            try:
                created_response = llm_client.responses.create(
                    model=ai_model,
                    input=prompt,
                    tools=[{"type": "web_search_preview"}],
                    timeout=id_timeout,
                    background=True,
                )

                print(f"Created response: {created_response.id}")
                return created_response.id

            except (APITimeoutError, APIConnectionError) as e:
                last_err = e
                print(f"Create failed (attempt {attempt}/{max_retry}): {e}")
                attempt += 1

        print("Failed to create reasoning job.")

        return {
            "status": "failed",
            "response": None,
            "error": str(last_err) if last_err else "Unable to create job",
            "response_id": None,
        }

    resp_id = get_resp_id()

    # get_resp_id now returns either an id string or a failure dict
    if isinstance(resp_id, dict):
        return resp_id

    last_status = None
    poll_errors = 0

    while True:

        try:
            print("Polling for response...")
            current_response = llm_client.responses.retrieve(resp_id)
            print("Successfully retrieved response status.")
            poll_errors = 0

        except (APITimeoutError, APIConnectionError) as e:
            poll_errors += 1

            print(
                f"Polling error ({poll_errors}/{max_poll_errors}): "
                f"{e}. Retrying in {poll_interval}s."
            )

            if poll_errors >= max_poll_errors:
                return {
                    "status": "failed",
                    "response": None,
                    "error": f"Exceeded max polling errors: {e}",
                    "response_id": resp_id,
                }

            time.sleep(poll_interval)
            continue

        if current_response.status != last_status:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"status={current_response.status}"
            )
            last_status = current_response.status

        if current_response.status == "completed":

            if getattr(current_response, "output_text", None):
                return {
                    "status": "success",
                    "response": current_response.output_text,
                    "error": None,
                    "response_id": resp_id,
                }

            print(
                "Completed but output_text was empty. "
                "Returning full response object."
            )

            return {
                "status": "failed",
                "response": current_response,
                "error": "Completed with no output_text",
                "response_id": resp_id,
            }

        if current_response.status in {
            "failed",
            "cancelled",
            "expired",
            "incomplete",
        }:
            return {
                "status": "failed",
                "response": current_response,
                "error": getattr(current_response, "error", None),
                "response_id": resp_id,
            }

        print(
            f"Still processing "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"status={current_response.status} "
            f"sleeping {poll_interval}s..."
        )

        time.sleep(poll_interval)


def restart_pipeline(saves_location = os.path.join(os.getcwd(), "data", "runs")):
    """
    Print resume instructions for the most recently completed pipeline stage.

    Scans the saved pipeline run directory for `_done` marker files, selects
    the most recently modified marker, and prints the recommended next class
    to initialize in order to resume the ReadingMachine workflow.

    Parameters
    ----------
    saves_location : str, default=os.path.join(os.getcwd(), "data", "runs")
        Root directory containing saved pipeline stage folders.

    Returns
    -------
    str or None
        Returns a message if no `_done` files are found. Otherwise, prints
        resume instructions and returns None.

    Notes
    -----
    Completion is inferred from `_done` marker files stored inside
    stage-specific save directories. The latest completed stage is selected
    using the marker file modification time.

    The generated instructions assume the standard ReadingMachine stage
    directory names, such as `01_search_strings`, `08_insights`, and
    `09_clusters`.

    This helper covers the getlit and core corpus-reading workflow through
    clustering. Once synthesis has begun, progress within the summarization
    workflow should be checked separately using `SummaryState.status()`.
    """
    
    def gen_pipeline_step(latest_file_path):

        #Get the latest path - i.e. excluding _done - to give the path to the state files
        latest_path = os.path.dirname(latest_file_path)
        # Get the latest step name to pass to the pipeline steps dictionary to get the correct text for the user
        latest_step = os.path.basename(latest_path)
        
        pipeline_steps = {
        "01_search_strings": ("You have generated search strings and saved them in your corpus_state. "
                              "You should continue with retreieving academic literature. Initialize AcademicLit as follows:\n"
                              f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                              "getlit.AcademicLit(corpus_state = latest_corpus_state)"),
        "02_academic_lit": ("You have retrieved academic literature and added it to your corpus_state. "
                           "You should continue with processing the literature. Initialize AcademicLit as follows:\n"
                           f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                           "getlit.GreyLiterature(corpus_state = latest_corpus_state, llm_client=llm_client)"),
        "03_grey_lit": ("You have acquired the relevant grey literature and added it to your corpus_state. "
                        "You should continue with the next step. Initialize the next class as follows:\n"
                        f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                        "getlit.Literature(corpus_state = latest_corpus_state)"),
        "04_literature_deduped": ("You have deduplicated the literature and updated your corpus_state. "
                                  "You should continue by condutcing an ai assisted check of your literature. Initialize the next class as follows:\n"
                                  f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                                  "getlit.AiLiteratureCheck(corpus_state = latest_corpus_state, llm_client=llm_client)"),
        "05_ai_lit_check": ("You have completed the AI literature check and updated your corpus_state. "
                           "You should proceed to set up your download architecture for your papers. Initialize the next class as follows:\n"
                           f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                           "getlit.DownloadManager(corpus_state = latest_corpus_state)"),
        "06_download_manager": ("You have completed the getlit workflow: You have downloaded your papers and updated your corpus_state. "
                                "You should proceed to the core workflow. The first step is to ingest the full text of your papers and chunk them. "
                                "Initialize the next class as follows:\n"
                                f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                                "core.Ingestor(corpus_state = latest_corpus_state, llm_client=llm_client, ai_model='gpt-4o')"),
        "07_full_text_and_chunks": ("You have ingested the full text of your papers, confirmed metadata, and chunked them. You should proceed to generate insights. Initialize the next class as follows:\n"
                                    f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                                    "core.Insights(corpus_state = latest_corpus_state, llm_client=llm_client, ai_model='gpt-4o', paper_context=paper_context)"),
        "08_insights": ("You have generated insights from your papers. You should proceed to the next step. Initialize the next class as follows:\n"
                        f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                        "core.Clustering(corpus_state = latest_corpus_state, llm_client=llm_client, embedding_model='text-embedding-3-small')"),
        "09_clusters": ("You have clustered your insights. You should proceed to the next step. Initialize the next class as follows:\n"
                        f"latest_corpus_state = state.CorpusState.load(filepath = '{latest_path}')\n"
                        "core.Summarize(corpus_state=latest_corpus_state, llm_client=llm_client, ai_model=\"gpt-4o\", paper_output_length=10000).\n\n"
                        "NOTE: If you are already working in Summrize you can determine your position in the Summarize pipeline via: Summarize.status()")
        }
        
        # Call the dict to return the text
        return(pipeline_steps[latest_step])
    
    done_files = []
    # List all the files 
    for root, dirs, files in os.walk(saves_location):
        done_files.extend([os.path.join(root, file) for file in files if file == "_done"])
    
    if len(done_files) == 0:
        return("No steps of the pipeline have been completed. You should start from the beginning.")
    done_timestamps = [os.path.getmtime(file) for file in done_files]
    latest_idx = np.argmax(done_timestamps)
    latest_file = done_files[latest_idx]
    
    # Generate the pipeline dictionary with the correct file location
    latest_step = gen_pipeline_step(latest_file)
    # Print the text containing instructions for the latest step
    print(latest_step)


#------------DEDUPLICATION FUNCTIONS START/----------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison, matching, and hashing operations.

    Applies a lightweight normalization procedure that:

    - converts text to lowercase
    - collapses consecutive whitespace into a single space
    - removes leading and trailing whitespace

    Parameters
    ----------
    text : str
        Text to normalize.

    Returns
    -------
    str
        Normalized text. Non-string inputs return an empty string.

    Notes
    -----
    This normalization is intentionally minimal and is designed to reduce
    superficial differences that can affect text comparison, similarity
    matching, and deterministic hashing.

    The function does not remove punctuation, perform stemming, normalize
    Unicode characters, or apply any semantic transformations.

    Because normalization lowercases text and collapses whitespace, the
    transformation is lossy and should not be used when exact text
    preservation is required.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

def drop_exact_author_title_year(corpus_state: CorpusState) -> CorpusState:
    """
    Remove exact duplicate papers based on normalized author, title, and date.

    Creates a copy of the supplied CorpusState, constructs a normalized
    author-title-date key for each row in `insights`, and keeps the first
    record for each unique key. The deduplicated set of `paper_id` values is
    then used to filter both `insights` and, when populated, `full_text`.

    Parameters
    ----------
    corpus_state : CorpusState
        CorpusState containing an `insights` table with `paper_author`,
        `paper_title`, `paper_date`, and `paper_id` columns.

    Returns
    -------
    CorpusState
        A copied CorpusState with duplicate papers removed from `insights`
        and, if present, `full_text`.

    Notes
    -----
    The input CorpusState is not modified.

    Duplicate detection is based on a normalized concatenation of
    `paper_author`, `paper_title`, and `paper_date`. Normalization lowercases
    text, collapses whitespace, and strips leading and trailing spaces.

    The function assumes that records with identical normalized
    author-title-date keys refer to the same paper. It is intended as a
    conservative exact-deduplication step before fuzzier duplicate detection.

    Only `insights` and `full_text` are filtered. Other CorpusState tables,
    such as `chunks`, are not modified.
    """
    # Make working copy
    temp_corpus_state = corpus_state.copy()
    df = temp_corpus_state.insights.copy()
    # Create author_title_year string for exact matching
    df["author_title_year"] = (
        df[["paper_author", "paper_title", "paper_date"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    # Normalize the string to avoid false negatives due to minor formatting differences
    df["norm_author_title_year"] = df["author_title_year"].apply(normalize_text)
    # Drop dupes
    df = df.drop_duplicates(subset="norm_author_title_year", keep="first")

    # Get unique IDs after dropping exact dupes to clean state objects
    valid_paper_ids = set(df["paper_id"])

    # Update the insights in corpus_state to only include the deduped paper ids
    temp_corpus_state.insights = temp_corpus_state.insights[temp_corpus_state.insights["paper_id"].isin(valid_paper_ids)].reset_index(drop=True)
    
    # Update full text if it needs it:
    if temp_corpus_state.full_text.shape[0] > 0:
        temp_corpus_state.full_text = temp_corpus_state.full_text[temp_corpus_state.full_text["paper_id"].isin(valid_paper_ids)].reset_index(drop=True)
    
    return temp_corpus_state


def drop_exact_hash(corpus_state: CorpusState) -> CorpusState:
    """
    Remove exact duplicate documents using normalized full-text hashes.

    Creates a copy of the supplied CorpusState, normalizes each document's
    full text, computes an MD5 hash of the normalized content, and removes
    documents with duplicate hashes. The resulting set of unique `paper_id`
    values is then used to filter both `full_text` and `insights`.

    Parameters
    ----------
    corpus_state : CorpusState
        CorpusState containing populated `full_text` and `insights` tables.

    Returns
    -------
    CorpusState
        A copied CorpusState containing only unique documents based on
        normalized full-text content.

    Notes
    -----
    The input CorpusState is not modified.

    Duplicate detection is based on the normalized document text rather than
    metadata. Documents with identical normalized text will be treated as
    duplicates regardless of title, author, or publication metadata.

    Text normalization is performed using `normalize_text()`, which
    lowercases text, collapses whitespace, and removes leading and trailing
    spaces before hashing.

    MD5 is used as a fast content-fingerprinting mechanism. While MD5 is not
    cryptographically secure, collision risk is negligible for document
    deduplication purposes.

    Only `insights` and `full_text` are filtered. Other CorpusState tables,
    such as `chunks`, are not modified.
    """
    # Hash function
    def hash_text(text:str) -> str:
        """
        Generate a stable hash for a document.

        Uses MD5 for speed (sufficient for deduplication purposes).
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()
   
    temp_corpus_state = corpus_state.copy()
    # Build hashes for the full text
    temp_full_text = temp_corpus_state.full_text.copy()
    # normalize the full text to avoid false negatives due to minor formatting differences
    temp_full_text["normalized"] = temp_full_text["full_text"].apply(normalize_text)
    # hash the text
    temp_full_text["paper_hash"] = temp_full_text["normalized"].apply(hash_text)

    # Drop duplicates based on the hash
    temp_full_text = temp_full_text.drop_duplicates(subset="paper_hash", keep="first").reset_index(drop=True)

    # Clean the state so that only valid deduped ids exist in all dfs
    valid_paper_ids = set(temp_full_text["paper_id"])
    # update the insights and full text to reflect only the deduped paper ids
    temp_corpus_state.insights = temp_corpus_state.insights[temp_corpus_state.insights["paper_id"].isin(valid_paper_ids)].reset_index(drop=True)
    temp_corpus_state.full_text = temp_corpus_state.full_text[temp_corpus_state.full_text["paper_id"].isin(valid_paper_ids)].reset_index(drop=True)
    
    return temp_corpus_state      

def gen_shingles_items(df: pd.DataFrame, k: int = 5) -> Dict[str, set]:
    """
    Generate k-word shingles for a collection of documents.

    Normalizes each document's full text and converts it into a set of
    overlapping k-word sequences ("shingles"). The resulting shingle sets can
    be used for document-similarity calculations such as Jaccard similarity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:

        - `paper_id`
        - `full_text`

    k : int, default=5
        Number of words per shingle.

    Returns
    -------
    dict[str, set[str]]
        Dictionary mapping each `paper_id` to its set of k-word shingles.

    Notes
    -----
    Text is normalized using `normalize_text()` before shingle generation.

    Documents with fewer than `k` words produce empty shingle sets.

    Shingles preserve local phrase structure and are commonly used for
    near-duplicate detection because they are more robust to minor formatting
    changes than exact text matching.

    This function is used as a preprocessing step for content-based document
    deduplication and similarity comparison.
    """

    def gen_shingles(text: str, k: int) -> set:
        """
        Generate shingles (k-grams) for a given text.
        """
        if not text:
            return set()
        words = text.split()
        if len(words) < k:
            return set()
        
        return {
            " ".join(words[i:i+k])
            for i in range(len(words) - k + 1)
        }
    
    temp_df = df.copy()
    # compute shingles for the full text
    temp_df["normalized_full_text"] = temp_df["full_text"].apply(normalize_text)
    temp_df["shingles"] = temp_df["normalized_full_text"].apply(lambda x: gen_shingles(x, k))

    # Create dict of paper id to shingles
    paper_ids = temp_df["paper_id"].tolist()
    shingles = temp_df["shingles"].tolist()
    return dict(zip(paper_ids, shingles))


def gen_title_items(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate normalized title representations for document comparison.

    Normalizes paper titles and returns a mapping from `paper_id` to the
    normalized title string. The resulting dictionary is intended for use in
    title-based similarity matching and duplicate-candidate generation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:

        - `paper_id`
        - `paper_title`

    Returns
    -------
    dict[str, str]
        Dictionary mapping each `paper_id` to its normalized title.

    Notes
    -----
    Titles are normalized using `normalize_text()`, which lowercases text,
    collapses whitespace, and removes leading and trailing spaces.

    Only title information is used. Author names, publication dates, and
    full-text content are intentionally excluded.

    Title-based matching is computationally inexpensive and useful for
    identifying potential duplicates before document ingestion or full-text
    processing. However, it is less precise than full-text similarity
    methods and is generally best used as a candidate-generation step rather
    than a final deduplication criterion.
    """
    temp_df = df.copy()
    temp_df["normalized_title"] = temp_df["paper_title"].apply(normalize_text)
    return dict(zip(temp_df["paper_id"], temp_df["normalized_title"]))

def jaccard_sim(set_a: set, set_b: set) -> float:
    """
    Compute the Jaccard similarity between two sets.

    Jaccard similarity measures the proportion of shared elements between
    two sets and is defined as:

        |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set_a : set
        First set.

    set_b : set
        Second set.

    Returns
    -------
    float
        Jaccard similarity score in the range [0.0, 1.0].

        - 1.0 indicates identical sets.
        - 0.0 indicates no shared elements.
        - Intermediate values indicate partial overlap.

    Notes
    -----
    If both sets are empty, the union is empty and the function returns
    0.0 by convention.

    This function is commonly used with document shingles for near-duplicate
    detection, where higher scores indicate greater overlap in phrase-level
    content.
    """
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def fuzzy_sim(str_a: str, str_b: str, scorer=fuzz.token_set_ratio) -> float:
    """
    Compute fuzzy string similarity using a RapidFuzz scorer.

    Applies the supplied RapidFuzz scoring function to two strings and
    returns the resulting similarity score.

    Parameters
    ----------
    str_a : str
        First string.

    str_b : str
        Second string.

    scorer : callable, default=fuzz.token_set_ratio
        RapidFuzz scoring function accepting two strings and returning a
        similarity score.

    Returns
    -------
    float
        Similarity score produced by the selected scorer. The score range
        depends on the scorer used. For `fuzz.token_set_ratio`, values
        typically range from 0 to 100.

    Notes
    -----
    The default scorer, `fuzz.token_set_ratio`, is tolerant of differences
    in word order and extra tokens, making it useful for title-based and
    metadata-based duplicate detection.

    This function is a lightweight wrapper around RapidFuzz and performs no
    additional normalization. Callers should normalize text before
    comparison when consistent formatting is required.
    """
    score = scorer(str_a, str_b)
    return score
    

def get_similar_groups(items: Dict[str, Any], similarity_fn, threshold: float) -> pd.DataFrame:
    """
    Group items whose pairwise similarity meets or exceeds a threshold.

    Builds an undirected graph in which each item ID is a node and an edge is
    added between two IDs when their similarity score is greater than or
    equal to `threshold`. Connected components in this graph are then
    returned as similarity groups.

    Parameters
    ----------
    items : dict[str, Any]
        Mapping from item IDs to comparable values. In ReadingMachine
        deduplication workflows, keys are typically `paper_id` values and
        values are normalized titles, shingle sets, or other comparison
        representations.

    similarity_fn : callable
        Function used to compare two item values. Must accept two values from
        `items` and return a numeric similarity score.

    threshold : float
        Minimum similarity score required to place an edge between two items.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per item and the columns:

        - `paper_id`: item ID from the input dictionary
        - `sim_group`: connected-component group ID, or `-1` for singleton
        items that have no similar matches

    Notes
    -----
    Similarity groups are transitive because they are based on connected
    components. For example, if A is similar to B and B is similar to C, all
    three items will be assigned to the same group even if A and C do not
    meet the threshold directly.

    The function performs all pairwise comparisons, so runtime scales
    quadratically with the number of items.

    Group IDs are assigned from the order of connected components returned
    by NetworkX and should not be treated as stable identifiers across runs
    or input order changes.
    """   
    # Create a graph and add all ids as nodes
    G = nx.Graph()
    G.add_nodes_from(items.keys())
    
    # Get the combinations i want to compare
    for id_a, id_b in combinations(items.keys(), 2):
    # Compute the similarity score between the two items
        score = similarity_fn(items[id_a], items[id_b])
    # Conditionally add to the graph
        if score >= threshold:
            G.add_edge(id_a, id_b)

    # Get all connected components - singletons have no edges and therfore are not a group
    components = list(nx.connected_components(G))

    # Create a dataframe with paper_id and sim_group, where sim_group is the group id of similar papers, and -1 for singletons
    groups = []

    for group_id, comp in enumerate(components, start=1):
        for paper_id in comp:
            groups.append({
                "paper_id": paper_id,
                "sim_group": group_id if len(comp) > 1 else -1
            })

    groups_df = pd.DataFrame(groups)

    return groups_df


def prepare_dedup_review(state: CorpusState, threshold: float, engine:str) -> pd.DataFrame:
    """
    Prepare candidate duplicate records for manual review.

    Creates a copied CorpusState, removes exact duplicates using the selected
    deduplication strategy, computes pairwise similarity groups, and returns
    an insights-based review table with similarity group assignments.

    Parameters
    ----------
    state : CorpusState
        CorpusState containing the corpus records to review.

    threshold : float
        Minimum similarity score required to group records as candidate
        duplicates. Expected scale depends on `engine`:

        - `"shingles"`: Jaccard similarity, typically in the range 0.0 to 1.0
        - `"fuzzy"`: RapidFuzz similarity, typically in the range 0 to 100

    engine : str
        Similarity method to use:

        - `"shingles"`: remove exact full-text duplicates, generate
        full-text shingles, and group records using Jaccard similarity
        - `"fuzzy"`: remove exact author-title-year duplicates, normalize
        titles, and group records using fuzzy title similarity

    Returns
    -------
    pd.DataFrame
        DataFrame containing the deduplicated `insights` rows plus a
        `sim_group` column. Candidate duplicate groups receive a positive
        group ID; singleton records receive `-1`.

    Raises
    ------
    ValueError
        If `engine` is not `"shingles"` or `"fuzzy"`.

    Notes
    -----
    The input CorpusState is not modified.

    This function prepares candidate duplicates for human review. It does
    not automatically remove near-duplicates.

    Results are sorted by `sim_group` in descending order so grouped
    candidate duplicates appear together near the top of the review table.

    The `"shingles"` engine requires populated `full_text` and is generally
    more precise because it compares document content. The `"fuzzy"` engine
    uses titles only and is useful before full-text ingestion, but is broader
    and noisier.
    """
    if engine not in ["shingles", "fuzzy"]:
        raise ValueError("engine must be either 'shingles' or 'fuzzy'")
    
    # Create a working copy
    temp_corpus_state = state.copy()
    
    # Process either on shingles or titles based on the engine choice
    if engine == "shingles":
        print("Dropping exact duplicates...")
        no_exact_dupes_state = drop_exact_hash(temp_corpus_state)
        print("Generating shingles...")
        items = gen_shingles_items(no_exact_dupes_state.full_text, k=5)
        similarity_fn = jaccard_sim

    else:
        print("Dropping exact duplicates...")
        no_exact_dupes_state = drop_exact_author_title_year(temp_corpus_state)
        print("Generating comparison strings...")
        items = gen_title_items(no_exact_dupes_state.insights)
        similarity_fn = fuzzy_sim

    print("Computing similar groups...")
    similar_groups_df = get_similar_groups(items=items, similarity_fn=similarity_fn, threshold=threshold)

    # Prepare df for manual review
    manual_review_df = (
        no_exact_dupes_state.insights
        .merge(
            similar_groups_df, 
            how="left", 
            on="paper_id"
            )
        .sort_values(by=["sim_group"], ascending=False)
        .reset_index(drop=True)
    )

    return manual_review_df

    
# --------------------DEDUPLICATION FUNCTIONS END/----------------------


def concat_with_schema(df1: pd.DataFrame, df2: pd.DataFrame, schema_from: str) -> pd.DataFrame:
    """
    Concatenate two DataFrames using one DataFrame as the schema reference.

    Aligns the non-reference DataFrame to the columns and dtypes of the
    reference DataFrame before concatenation. Missing columns are created
    with the reference dtype, columns are reordered to match the reference,
    and existing columns are cast to the reference schema.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame.

    df2 : pd.DataFrame
        Second DataFrame.

    schema_from : str
        Which DataFrame should provide the output schema. Must be one of:

        - `"top"`: use `df1` as the schema reference and return `df1`
        followed by aligned `df2`
        - `"bottom"`: use `df2` as the schema reference and return aligned
        `df1` followed by `df2`

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame whose columns and dtypes follow the selected
        reference schema.

    Raises
    ------
    ValueError
        If `schema_from` is not `"top"` or `"bottom"`.

    Notes
    -----
    Columns present only in the non-reference DataFrame are dropped, because
    the output is restricted to the reference schema.

    This helper exists to make concatenation stable across runs when one side
    may be empty or missing columns. Missing columns are created with the
    target dtype directly rather than being filled with `NaN` and cast later.
    """
    if schema_from == "top":
        ref = df1
        other = df2.copy()
    elif schema_from == "bottom":
        ref = df2
        other = df1.copy()
    else:
        raise ValueError("schema_from must be 'top' or 'bottom'")

    # Add missing columns with correct dtype
    for col, dtype in ref.dtypes.items():
        if col not in other.columns:
            other[col] = pd.Series(index=other.index, dtype=dtype)

    # Reorder columns
    other = other[ref.columns]

    # Cast existing columns
    other = other.astype(ref.dtypes.to_dict(), copy=False)

    # Concat
    if schema_from == "top":
        return pd.concat([ref, other], ignore_index=True)
    else:
        return pd.concat([other, ref], ignore_index=True)


def safe_pickle(obj, path, retries=6, base_delay=0.05, backoff=2.0):
    """
    Persist an object to disk using atomic file replacement with retry logic.

    Serializes an object using pickle, writes it to a temporary file, flushes
    the file contents to disk, and then atomically replaces the target file
    using `os.replace()`. If the replacement fails due to a transient
    `PermissionError`, the operation is retried using exponential backoff.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    path : str
        Destination file path.

    retries : int, default=6
        Maximum number of write/replace attempts before raising an error.

    base_delay : float, default=0.05
        Initial delay between retry attempts, in seconds.

    backoff : float, default=2.0
        Multiplicative factor applied to the retry delay after each failed
        attempt.

    Returns
    -------
    None

    Raises
    ------
    PermissionError
        If all retry attempts fail due to file-access issues.

    RuntimeError
        If the function reaches an unexpected failure state after exhausting
        retries.

    Notes
    -----
    The object is first written to a uniquely named temporary file and then
    moved into place with `os.replace()`. This helps prevent partially
    written pickle files from being observed by other processes.

    File contents are explicitly flushed and synchronized to disk using
    `flush()` and `os.fsync()` before replacement.

    Retry logic only handles `PermissionError`. Any other exception raised
    during serialization or file operations is propagated immediately.
    """
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    delay = base_delay

    for attempt in range(retries):
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(obj, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, path)
            return

        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff

    # Should never reach here
    raise RuntimeError("safe_pickle failed unexpectedly")

def sample_to_word_limit(
    texts,
    max_words: int = 70000,
    seed: int = config.seed
):
    """
    Randomly sample text items up to a cumulative word-count limit.

    Shuffles the input collection and selects items sequentially until
    adding the next item would exceed `max_words`. The result is a random
    subset whose total word count remains within the specified budget.

    Parameters
    ----------
    texts : Sequence[str]
        Collection of text items to sample from.

    max_words : int, default=70000
        Maximum cumulative word count permitted across the selected items.

    seed : int, default=config.seed
        Random seed used to initialize the shuffle for reproducible
        sampling.

    Returns
    -------
    list[str]
        Randomly ordered subset of the input texts whose combined word count
        does not exceed `max_words`.

    Notes
    -----
    Sampling is performed by shuffling the input and then taking the longest
    prefix that satisfies the word-count constraint.

    The original order of `texts` is not preserved.

    Word count is estimated using `len(text.split())`.

    This function is used to constrain prompt size when a complete insight
    set exceeds the practical input limits of a synthesis step. In the
    ReadingMachine workflow, any resulting omissions are expected to be
    recovered through downstream orphan detection, reinsertion, and
    iterative re-theming.

    Because selection is prefix-based after shuffling, inclusion
    probabilities are not identical to sampling without replacement under a
    fixed-size design. The function prioritizes respecting the word budget
    rather than producing a statistically representative sample.
    """

    random.seed(seed)

    shuffled = texts.copy()
    random.shuffle(shuffled)

    selected = []
    running_words = 0

    for text in shuffled:
        w = len(text.split())

        if running_words + w > max_words:
            break

        selected.append(text)
        running_words += w

    return selected
