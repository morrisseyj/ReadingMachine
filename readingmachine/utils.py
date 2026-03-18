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

import ast
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
import json
import pyarrow as pa
import os
import datetime
import time
from openai import OpenAI, APITimeoutError, APIConnectionError
import networkx as nx
from rapidfuzz import process, fuzz

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
    

def call_chat_completion(llm_client, 
                         ai_model, 
                         sys_prompt, 
                         user_prompt, 
                         fall_back: Dict[str, Any], 
                         return_json: bool, 
                         json_schema = None):
    """
    Standardized wrapper for chat completion calls.

    This function provides a consistent interface for interacting with
    chat completion models throughout the codebase. It handles both
    structured JSON responses and plain text outputs.

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
        invalid output.

    return_json : bool
        If True, the function attempts to parse the model response
        as JSON.

    json_schema : Optional[dict]
        Optional JSON schema used to enforce structured responses.

    Returns
    -------
    Union[str, Dict[str, Any]]
        Parsed JSON object if `return_json=True`, otherwise the raw
        text response.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if return_json:
        try:
            if json_schema is not None:
                response = llm_client.chat.completions.create(
                    model=ai_model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema
                    },
                    temperature=0
                )
            else:
                response = llm_client.chat.completions.create(
                    model=ai_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0
                )
        except Exception as e:
            print(f"Call to OpenAI failed. Error: {e}")
            return fall_back

        try:
            parsed = json.loads(response.choices[0].message.content.strip())
            return parsed
        except Exception as e:
            print(f"LLM failed to return valid JSON: {e}")
            return fall_back

    else:
        # -------- TEXT MODE --------
        try:
            response = llm_client.chat.completions.create(
                model=ai_model,
                messages=messages, 
                temperature=0
            )
        except Exception as e:
            print(f"Call to OpenAI failed. Error: {e}")
            return ""

        # just return raw text
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
                                    "core.InsightsGenerator(corpus_state = latest_corpus_state, llm_client=llm_client)"),
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


def gen_duplicate_check_string(df: pd.DataFrame) -> pd.Series:
    """
    Generate a string used for duplicate checking.

    This string combines author names, paper title, and publication year
    into a single string for each record.

    Returns
    -------
    pd.Series
        Series containing the duplicate check strings.
    """
    for col in ["paper_author", "paper_title", "paper_date"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column for duplicate removal.")

    authors_str = df["paper_author"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    authors_str = authors_str.fillna("Unknown Authors")
    authors_str = authors_str.apply(lambda x: x if x.strip().lower() else "Unknown Authors")
    title_str = df["paper_title"].astype(str)
    date_str = df["paper_date"].astype(str)

    return authors_str + " " + title_str + " " + date_str

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate literature records.

    Exact duplicates are identified using a constructed
    `duplicate_check_string`, which combines author names,
    paper title, and publication year.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing literature records.

    Returns
    -------
    pd.DataFrame
        DataFrame with exact duplicates removed.

    Notes
    -----
    - This operation is deterministic.
    - The first occurrence of a duplicate is retained.
    - The helper column `duplicate_check_string` is removed before return.
    """

    df = df.copy()

    # Construct duplicate matching key
    df["duplicate_check_string"] = gen_duplicate_check_string(df)

    # Drop exact duplicates
    df = df.drop_duplicates(subset="duplicate_check_string", keep="first")

    # Clean up helper column
    df = df.drop(columns=["duplicate_check_string"])

    return df


def get_fuzzy_matches(
    df: pd.DataFrame,
    similarity_threshold: int,
) -> List[Tuple[str, str]]:
    """
    Identify potential duplicate records using fuzzy string matching.

    Computes pairwise similarity scores across all records using
    RapidFuzz and returns pairs exceeding the similarity threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing literature records.

    similarity_threshold : int, default=90
        Minimum similarity score required to flag a pair as a potential duplicate.

    Returns
    -------
    List[Tuple[str, str]]
        List of tuple pairs representing potentially duplicated records.

    Notes
    -----
    - Uses RapidFuzz `ratio` scorer (string similarity).
    - O(N^2) complexity — can become expensive for large corpora.
    """

    df = df.copy()

    # Generate matching strings
    df["duplicate_check_string"] = gen_duplicate_check_string(df)

    strings = df["duplicate_check_string"].tolist()

    # Compute pairwise similarity matrix
    similarity_matrix = process.cdist(strings, strings, scorer=fuzz.ratio)

    fuzzy_pairs: List[Tuple[str, str]] = []

    for i, row in enumerate(similarity_matrix):
        for j, score in enumerate(row):
            if i < j and score >= similarity_threshold:
                fuzzy_pairs.append((strings[i], strings[j]))

    print("Pairwise fuzzy similarity computed.")

    return fuzzy_pairs

def get_similar_groups(
    df: pd.DataFrame,
    fuzzy_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Group fuzzy duplicate matches into similarity clusters.

    Constructs a graph where nodes are records and edges represent
    high-similarity matches. Connected components define groups
    of potentially duplicate records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing literature records.

    fuzzy_pairs : List[Tuple[str, str]]
        List of pairwise duplicate matches from `get_fuzzy_matches`.

    Returns
    -------
    pd.DataFrame
        DataFrame mapping each record to a similarity group.

        Columns:
        - duplicate_check_string
        - sim_group (int)

    Notes
    -----
    - Records without matches are assigned `sim_group = -1`.
    - Groups represent *candidate duplicates*, not confirmed duplicates.
    """

    df = df.copy()

    if "duplicate_check_string" not in df.columns:
        df["duplicate_check_string"] = gen_duplicate_check_string(df)

    # Build graph
    graph = nx.Graph()
    graph.add_edges_from(fuzzy_pairs)

    # Identify connected components
    components = list(nx.connected_components(graph))

    grouped_records = []
    matched_strings = set()

    # Assign group IDs
    for group_id, component in enumerate(components, start=1):
        for string in component:
            grouped_records.append({
                "duplicate_check_string": string,
                "sim_group": group_id
            })
            matched_strings.add(string)

    # Assign unmatched records
    for string in df["duplicate_check_string"]:
        if string not in matched_strings:
            grouped_records.append({
                "duplicate_check_string": string,
                "sim_group": -1
            })

    groups_df = pd.DataFrame(grouped_records)

    return groups_df

def prepare_fuzzy_review_df(
    df: pd.DataFrame,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Prepare a DataFrame for manual duplicate review.

    This function:
    1. Generates fuzzy matches
    2. Groups them into similarity clusters
    3. Merges results back with original data

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing literature records.

    similarity_threshold : int, default=90
        Threshold for fuzzy matching.

    Returns
    -------
    pd.DataFrame
        DataFrame suitable for manual inspection.

    Notes
    -----
    - Output should be exported (CSV/Excel) for manual review.
    - Users should delete duplicate rows manually.
    """

    df = df.copy()

    # Step 1: generate matching strings
    df["duplicate_check_string"] = gen_duplicate_check_string(df)

    # Step 2: fuzzy match pairs
    fuzzy_pairs = get_fuzzy_matches(df, similarity_threshold)

    # Step 3: group matches
    groups_df = get_similar_groups(df, fuzzy_pairs)

    # Step 4: merge back for inspection
    review_df = groups_df.merge(
        df,
        how="left",
        on="duplicate_check_string"
    )

    return review_df







