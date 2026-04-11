"""
Utility functions used across the ReadingMachine codebase.

This module provides shared helpers that support the core pipeline,
rendering layer, and corpus discovery tools. The utilities include:

    - pipeline state validation
    - standardized LLM API calls
    - reasoning-model orchestration
    - pipeline restart assistance

These functions centralize repeated logic so that the rest of the
codebase can remain focused on the analytical workflow rather than
infrastructure details.

Design principles
-----------------

The utilities are designed to:

    • enforce consistent input validation across modules
    • standardize LLM interactions
    • support resumable execution of long-running operations
    • assist users in recovering pipeline progress

None of the utilities modify the analytical logic of ReadingMachine;
they exist solely to support reliable execution of the pipeline.
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

def validate_format(
    corpus_state: Optional["CorpusState"] = None, 
    questions: Optional[pd.DataFrame] = None,
    injected_value: Optional[pd.DataFrame] = None,
    state_required_cols: List[str] = [],
    injected_required_cols: List[str] = []
) -> "CorpusState":
    """
   Validate and normalize pipeline input state.

    This helper ensures that downstream classes receive a properly
    structured `CorpusState` object regardless of how the input
    was provided.

    The function supports two initialization paths:

        Path A — Existing CorpusState
            Validate the structure of the provided state object.

        Path B — Injected DataFrame
            Construct a new `CorpusState` from a questions DataFrame
            and an injected dataset.

    Parameters
    ----------
    corpus_state : Optional[CorpusState]
        Existing state object to validate.

    questions : Optional[pd.DataFrame]
        Question metadata used to initialize a new state.

    injected_value : Optional[pd.DataFrame]
        DataFrame containing records to insert into the new state.

    state_required_cols : List[str]
        Columns required to exist in `corpus_state.insights`.

    injected_required_cols : List[str]
        Columns required to exist in the injected DataFrame.

    Returns
    -------
    CorpusState
        Validated or newly constructed state object.

    Raises
    ------
    ValueError
        If the input arguments do not satisfy one of the valid
        initialization paths or if required columns are missing.
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
    ai_model, 
    sys_prompt, 
    user_prompt, 
    fall_back: Dict[str, Any], 
    return_json: bool, 
    json_schema=None,
    max_tokens: int | None = None
):
    """
    Standardized wrapper for chat completion calls.

    This function provides a consistent interface for interacting with
    chat completion models throughout the codebase. It supports both
    structured JSON responses and plain text outputs, and optionally
    allows control over the maximum number of tokens generated.

    The function centralizes:

        • prompt construction
        • response format handling (JSON vs text)
        • error handling and fallback behavior
        • optional output length control via max_tokens

    Parameters
    ----------
    llm_client : Any
        LLM API client instance.

    ai_model : str
        Model identifier used for the completion request.

    sys_prompt : str
        System prompt defining the model's role and instructions.

    user_prompt : str
        User prompt containing task-specific input.

    fall_back : Dict[str, Any]
        Default value returned if the LLM call fails or returns
        invalid output (only used when return_json=True).

    return_json : bool
        If True, the function attempts to parse the model response
        as JSON and return a dictionary. If False, raw text is returned.

    json_schema : Optional[dict], default=None
        Optional JSON schema used to enforce structured responses.
        If provided, the model is constrained to return output matching
        this schema.

    max_tokens : Optional[int], default=None
        Maximum number of tokens the model is allowed to generate.
        If None, the API default is used. This parameter should be used
        when strict output length control is required (e.g. to prevent
        truncation or enforce bounded summaries).

    Returns
    -------
    Union[str, Dict[str, Any]]
        Parsed JSON object if `return_json=True`, otherwise the raw
        text response string.

    Notes
    -----
    - If `return_json=True` and the model output cannot be parsed as valid JSON,
    the function returns `fall_back`.
    - If `return_json=False`, failures return an empty string.
    - Setting `max_tokens` does not reduce the model's input capacity; it only
    constrains output length and helps prevent incomplete or truncated responses.
    - For tasks requiring strict output bounds (e.g. orphan integration),
    it is recommended to set `max_tokens` slightly above the expected
    output length to allow for formatting overhead.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    kwargs = {
        "model": ai_model,
        "messages": messages,
        "temperature": 0
    }

    if return_json:
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = llm_client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Call to OpenAI failed. Error: {e}")
        return fall_back if return_json else ""

    if return_json:
        try:
            parsed = json.loads(response.choices[0].message.content.strip())
            return parsed
        except Exception as e:
            print(f"LLM failed to return valid JSON: {e}")
            return fall_back
    else:
        return response.choices[0].message.content

def call_reasoning_model(
    prompt: str,
    llm_client: OpenAI,
    ai_model: str = "o3-deep-research",
    id_timeout: float = 30,
    resp_timeout: float = 1500,
    max_retry: int = 2,
):
    """
    Execute a reasoning-model job with asynchronous polling.

    This function manages long-running LLM reasoning tasks that may
    require extended processing time. The request is submitted in
    background mode and polled until completion.

    Parameters
    ----------
    prompt : str
        Prompt sent to the reasoning model.

    llm_client : OpenAI
        OpenAI client instance.

    ai_model : str
        Reasoning model identifier.

    id_timeout : float
        Timeout (seconds) for obtaining the job identifier.

    resp_timeout : float
        Maximum allowed wait time for job completion.

    max_retry : int
        Maximum number of retries when creating the job.

    Returns
    -------
    dict
        Dictionary containing:

            status : "success" or "failed"
            response : model output text or response object
    """
    # Get a response id with background=True
    def get_resp_id():
        attempt = 1
        last_err = None
        while attempt <= max_retry:
            try:
                resp = llm_client.responses.create(
                    model=ai_model,
                    input=prompt,
                    tools=[{"type": "web_search_preview"}],
                    timeout=id_timeout,
                    background=True,
                )
                return resp.id
            except (APITimeoutError, APIConnectionError) as e:
                last_err = e
                print(f"Create failed (attempt {attempt}/{max_retry}): {e}")
                attempt += 1
        print("Failed to create deep-research job.")
        if last_err:
            print(f"Last error: {last_err}")
        return None

    resp_id = get_resp_id()
    if resp_id is None:
        print("Could not obtain response ID; aborting.")
        output = {"status": "failed", "response": None}
        return output

    end_time = time.time() + resp_timeout
    last_status = None

    while True:
        if time.time() > end_time:
            print(f"Max wait time ({resp_timeout}s) exceeded.")
            return None

        try:
            resp = llm_client.responses.retrieve(resp_id)
        except (APITimeoutError, APIConnectionError) as e:
            print(f"Polling error: {e}; retrying in 10s.")
            time.sleep(10)
            continue

        if resp.status != last_status:
            print(f"Status: {resp.status}")
            last_status = resp.status

        if resp.status == "completed":
            # Prefer the convenience field
            if getattr(resp, "output_text", None):
                output = {"status": "success", "response": resp.output_text}
                return output
            # Fallback: return the full object so you can inspect
            print("Completed with no output_text; returning raw response object.")
            output = {"status": "failed", "response": resp}
            return output

        if resp.status == "failed":
            print("Deep research failed.")
            print("Error:", getattr(resp, "error", None))
            return None

        print("Still processing; sleeping 60s...")
        time.sleep(60)


def restart_pipeline(saves_location = os.path.join(os.getcwd(), "data", "runs")):

    """
     Identify the latest completed pipeline stage and provide recovery instructions.

    This helper scans the pipeline run directory for `_done` marker files
    indicating completed stages. It identifies the most recent stage and
    prints instructions for resuming the pipeline from that point.

    Parameters
    ----------
    saves_location : str
        Directory containing saved pipeline state folders.

    Notes
    -----
    This utility is intended to help users recover progress in long
    pipelines without manually inspecting state directories.
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
    done_timestamps = [os.path.getmtime(os.path.join(root, file)) for file in done_files]
    latest_idx = np.argmax(done_timestamps)
    latest_file = done_files[latest_idx]
    
    # Generate the pipeline dictionary with the correct file location
    latest_step = gen_pipeline_step(latest_file)
    # Print the text containing instructions for the latest step
    print(latest_step)


#------------DEDUPLICATION FUNCTIONS START/----------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for downstream comparison and hashing.

    This function standardizes text by:
    - converting to lowercase
    - collapsing consecutive whitespace
    - stripping leading/trailing spaces

    Non-string inputs return an empty string.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Normalized text.

    Notes
    -----
    - Used to reduce false negatives in both hashing and similarity comparisons.
    - Does not remove punctuation (handled elsewhere if needed).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

def drop_exact_author_title_year(corpus_state: CorpusState) -> CorpusState:
    """
    Remove exact duplicate records based on author, title, and publication date.

    Constructs a normalized composite string of (author + title + date) and
    removes duplicate entries using exact matching. Updates both `insights`
    and `full_text` to retain only the deduplicated set of paper_ids.

    Parameters
    ----------
    corpus_state : CorpusState
        Input state containing `insights` and optionally `full_text`.

    Returns
    -------
    CorpusState
        New CorpusState with duplicates removed.

    Notes
    -----
    - Operates on a copy of the input state (does not mutate original).
    - Intended as a fast, low-risk pre-filter before fuzzy matching.
    - Assumes that identical author-title-date combinations are duplicates.
    """
    # Make working copy
    temp_corpus_state = corpus_state.copy()
    df = temp_corpus_state.insights.copy()
    # Create author_title_year string for exact matching
    df["author_title_year"] = df["paper_author"].fillna("") + " " + df["paper_title"].fillna("") + " " + df["paper_date"].fillna("")
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
    Remove exact duplicate documents using a hash of normalized full text.

    Computes an MD5 hash of normalized document text and removes duplicates
    based on identical hashes. Updates both `insights` and `full_text` to
    retain only unique documents.

    Parameters
    ----------
    corpus_state : CorpusState
        Input state containing `full_text` and `insights`.

    Returns
    -------
    CorpusState
        New CorpusState with exact duplicate documents removed.

    Notes
    -----
    - Uses normalized text to avoid false negatives due to formatting differences.
    - MD5 is used for speed; collision risk is negligible for this use case.
    - Operates on a copy of the input state.
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
    Generate k-word shingles for each document in the DataFrame.

    Each document is converted into a set of overlapping k-word sequences
    ("shingles"), which are later used for Jaccard similarity comparison.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:
        - 'paper_id'
        - 'full_text'

    k : int, default=5
        Number of words per shingle.

    Returns
    -------
    Dict[str, set]
        Dictionary mapping paper_id → set of shingles.

    Notes
    -----
    - Empty or very short texts return empty sets.
    - Shingles capture phrase-level structure, making them robust to formatting changes.
    - Used for full-text near-duplicate detection.
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
    Generate normalized title representations for fuzzy matching.

    Each paper_id is mapped to a normalized version of its title,
    which is used for fuzzy similarity comparison.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - 'paper_id'
        - 'paper_title'

    Returns
    -------
    Dict[str, str]
        Dictionary mapping paper_id → normalized title string.

    Notes
    -----
    - Only title is used (author/year intentionally excluded).
    - Designed to surface candidate duplicates for manual review.
    - Less precise than full-text matching but useful pre-ingestion.
    """
    temp_df = df.copy()
    temp_df["normalized_title"] = temp_df["paper_title"].apply(normalize_text)
    return dict(zip(temp_df["paper_id"], temp_df["normalized_title"]))

def jaccard_sim(set_a: set, set_b: set) -> float:
    """
    Compute Jaccard similarity between two sets.

    Jaccard similarity is defined as:
        |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set_a : set
        First set (e.g., shingles of document A).

    set_b : set
        Second set (e.g., shingles of document B).

    Returns
    -------
    float
        Similarity score in [0, 1].

    Notes
    -----
    - Returns 0.0 if both sets are empty.
    - High values (~0.9+) indicate near-identical documents.
    """
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def fuzzy_sim(str_a: str, str_b: str, scorer=fuzz.token_set_ratio) -> float:
    """
    Compute fuzzy similarity between two strings.

    Uses RapidFuzz scorers (default: token_set_ratio) to measure
    similarity between two text strings.

    Parameters
    ----------
    str_a : str
        First string.

    str_b : str
        Second string.

    scorer : callable, default=fuzz.token_set_ratio
        RapidFuzz scoring function.

    Returns
    -------
    float
        Similarity score (typically 0–100 for token_set_ratio).

    Notes
    -----
    - token_set_ratio is robust to word order and extra tokens.
    - Used for metadata-based deduplication (titles).
    """
    score = scorer(str_a, str_b)
    return score
    

def get_similar_groups(items: Dict[str, Any], similarity_fn, threshold: float) -> pd.DataFrame:
    """
    Compute fuzzy similarity between two strings.

    Uses RapidFuzz scorers (default: token_set_ratio) to measure
    similarity between two text strings.

    Parameters
    ----------
    str_a : str
        First string.

    str_b : str
        Second string.

    scorer : callable, default=fuzz.token_set_ratio
        RapidFuzz scoring function.

    Returns
    -------
    float
        Similarity score (typically 0–100 for token_set_ratio).

    Notes
    -----
    - token_set_ratio is robust to word order and extra tokens.
    - Used for metadata-based deduplication (titles).
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
    Prepare a DataFrame for manual deduplication review.

    This function:
    1. Removes exact duplicates (hash or author-title-year)
    2. Builds representations (shingles or titles)
    3. Computes similarity groups using graph-based clustering
    4. Returns a DataFrame for manual inspection

    Parameters
    ----------
    state : CorpusState
        Input corpus state containing `insights` and/or `full_text`.

    threshold : float
        Similarity threshold for grouping:
        - ~0.9 for shingles (Jaccard)
        - ~85 for fuzzy title matching

    engine : str
        Deduplication method:
        - "shingles" → full-text similarity
        - "fuzzy" → title similarity

    Returns
    -------
    pd.DataFrame
        DataFrame of candidate duplicates with columns:
        - original insights fields
        - 'sim_group'

    Notes
    -----
    - Does NOT mutate the original state.
    - Intended for human review (not automatic deletion).
    - Results are sorted by sim_group to cluster duplicates together.
    - Full-text method is more precise; fuzzy is broader but noisier.
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




# def gen_duplicate_check_string(df: pd.DataFrame) -> pd.Series:
#     """
#     Generate a string used for duplicate checking.

#     This string combines author names, paper title, and publication year
#     into a single string for each record.

#     Returns
#     -------
#     pd.Series
#         Series containing the duplicate check strings.
#     """
#     for col in ["paper_author", "paper_title", "paper_date"]:
#         if col not in df.columns:
#             raise ValueError(f"DataFrame must contain '{col}' column for duplicate removal.")

#     authors_str = df["paper_author"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
#     authors_str = authors_str.fillna("Unknown Authors")
#     authors_str = authors_str.apply(lambda x: x if x.strip().lower() else "Unknown Authors")
#     title_str = df["paper_title"].astype(str)
#     date_str = df["paper_date"].astype(str)

#     return authors_str + " " + title_str + " " + date_str

# def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Remove exact duplicate literature records.

#     Exact duplicates are identified using a constructed
#     `duplicate_check_string`, which combines author names,
#     paper title, and publication year.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing literature records.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with exact duplicates removed.

#     Notes
#     -----
#     - This operation is deterministic.
#     - The first occurrence of a duplicate is retained.
#     - The helper column `duplicate_check_string` is removed before return.
#     """

#     df = df.copy()

#     # Construct duplicate matching key
#     df["duplicate_check_string"] = gen_duplicate_check_string(df)

#     # Drop exact duplicates
#     df = df.drop_duplicates(subset="duplicate_check_string", keep="first")

#     # Clean up helper column
#     df = df.drop(columns=["duplicate_check_string"])

#     return df


# def get_fuzzy_matches(
#     df: pd.DataFrame,
#     similarity_threshold: int = 90,
# ) -> List[Tuple[str, str]]:
#     """
#     Identify potential duplicate records using fuzzy string matching.

#     Computes pairwise similarity scores across all records using
#     RapidFuzz and returns pairs exceeding the similarity threshold.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing literature records.

#     similarity_threshold : int, default=90
#         Minimum similarity score required to flag a pair as a potential duplicate.

#     Returns
#     -------
#     List[Tuple[str, str]]
#         List of tuple pairs representing potentially duplicated records.

#     Notes
#     -----
#     - Uses RapidFuzz `token_set_ratio` scorer (word similarity).
#     - O(N^2) complexity — can become expensive for large corpora.
#     """

#     df = df.copy()

#     # Generate matching strings
#     df["duplicate_check_string"] = gen_duplicate_check_string(df)

#     strings = df["duplicate_check_string"].tolist()

#     # Compute pairwise similarity matrix
#     similarity_matrix = process.cdist(strings, strings, scorer=fuzz.token_set_ratio)

#     fuzzy_pairs: List[Tuple[str, str]] = []

#     for i, row in enumerate(similarity_matrix):
#         for j, score in enumerate(row):
#             if i < j and score >= similarity_threshold:
#                 fuzzy_pairs.append((strings[i], strings[j]))

#     print("Pairwise fuzzy similarity computed.")

#     return fuzzy_pairs

# def get_similar_groups(
#     df: pd.DataFrame,
#     fuzzy_pairs: List[Tuple[str, str]]
# ) -> pd.DataFrame:
#     """
#     Group fuzzy duplicate matches into similarity clusters.

#     Constructs a graph where nodes are records and edges represent
#     high-similarity matches. Connected components define groups
#     of potentially duplicate records.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing literature records.

#     fuzzy_pairs : List[Tuple[str, str]]
#         List of pairwise duplicate matches from `get_fuzzy_matches`.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame mapping each record to a similarity group.

#         Columns:
#         - duplicate_check_string
#         - sim_group (int)

#     Notes
#     -----
#     - Records without matches are assigned `sim_group = -1`.
#     - Groups represent *candidate duplicates*, not confirmed duplicates.
#     """

#     df = df.copy()

#     if "duplicate_check_string" not in df.columns:
#         df["duplicate_check_string"] = gen_duplicate_check_string(df)

#     # Build graph
#     graph = nx.Graph()
#     graph.add_nodes_from(df["duplicate_check_string"])
#     graph.add_edges_from(fuzzy_pairs)

#     # Identify connected components
#     components = list(nx.connected_components(graph))

#     grouped_records = []
    
#     for group_id, component in enumerate(components, start=1):
#         for string in component:
#             grouped_records.append({
#                 "duplicate_check_string": string,
#                 "sim_group": group_id if len(component) > 1 else -1
#             })

#     groups_df = pd.DataFrame(grouped_records)

#     return groups_df

# def prepare_fuzzy_review_df(
#     df: pd.DataFrame,
#     similarity_threshold: int = 90
# ) -> pd.DataFrame:
#     """
#     Prepare a DataFrame for manual duplicate review.

#     This function:
#     1. Generates fuzzy matches
#     2. Groups them into similarity clusters
#     3. Merges results back with original data

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing literature records.

#     similarity_threshold : int, default=90
#         Threshold for fuzzy matching.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame suitable for manual inspection.

#     Notes
#     -----
#     - Output should be exported (CSV/Excel) for manual review.
#     - Users should delete duplicate rows manually.
#     """

#     df = df.copy()

#     # Step 1: generate matching strings
#     df["duplicate_check_string"] = gen_duplicate_check_string(df)

#     # Step 2: fuzzy match pairs
#     fuzzy_pairs = get_fuzzy_matches(df, similarity_threshold)

#     # Step 3: group matches
#     groups_df = get_similar_groups(df, fuzzy_pairs)

#     # Step 4: merge back for inspection
#     review_df = groups_df.merge(
#         df,
#         how="left",
#         on="duplicate_check_string"
#     )

#     # Sort to make manual review easier
#     review_df = review_df.sort_values(by=["sim_group"], ascending=False).reset_index(drop=True)

#     return review_df



def concat_with_schema(df1: pd.DataFrame, df2: pd.DataFrame, schema_from: str) -> pd.DataFrame:
    """
    Safely concatenate two DataFrames, ensuring schema consistency. Built because of pandas new behavior of not allowing astype on empty columns, 
    which was causing issues with our previous concat approach where we were filling missing columns with NaN and then trying to astype to ensure consistent dtypes across runs.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame.

    df2 : pd.DataFrame
        Second DataFrame.

    schema_from : str
        Source of the schema to enforce ("top" or "bottom").

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with consistent schema.
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
    Safely pickles an object to disk using atomic replace with retry + exponential backoff.

    Parameters
    ----------
    obj : Any
        The object to pickle.
    path : str
        The file path to save the pickle.
    retries : int
        Number of replace attempts before failing.
    base_delay : float
        Initial delay between retries (seconds).
    backoff : float
        Multiplier for exponential backoff.

    Returns
    -------
    None
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
    Randomly sample a list of text items so that the total word count
    does not exceed a specified limit.

    This function shuffles the input list and selects a prefix whose
    cumulative word count remains within `max_words`. It is designed
    to enforce input size constraints for LLM calls while preserving
    a representative random subset of the original data.

    Parameters
    ----------
    texts : List[str]
        List of text items (e.g. insights) to sample from.

    max_words : int
        Maximum total word count allowed across the selected texts.

    seed : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    List[str]
        Subset of input texts whose combined word count does not
        exceed `max_words`.

    Notes
    -----
    - Sampling is uniform over permutations (via shuffle).
    - Order is not preserved.
    - This is preferable to count-based sampling when model constraints
      depend on total input size rather than number of items.
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