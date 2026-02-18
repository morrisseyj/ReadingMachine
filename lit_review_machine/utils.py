from lit_review_machine.state import QuestionState

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

# def ensure_list_of_strings(val):
#     """
#     Normalize input to a list of strings.
#     Handles lists, strings, NaN, and other types.
#     Flattens list-of-lists if encountered.
#     """
#     # Flatten list-of-lists (e.g., [["Smith", "Jones"]])
#     if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
#         val = val[0]
#     if isinstance(val, list):
#         return [str(v) for v in val if v is not None]
#     if isinstance(val, str):
#         try:
#             parsed = ast.literal_eval(val)
#             if isinstance(parsed, list):
#                 return [str(v) for v in parsed if v is not None]
#         except Exception:
#             pass
#         return [val]
#     if pd.isna(val):
#         return []
#     return [str(val)]

def validate_format(
    state: Optional["QuestionState"], 
    questions: Optional[pd.DataFrame],
    injected_value: Optional[pd.DataFrame],
    state_required_cols: List[str], 
    injected_required_cols: List[str]
) -> "QuestionState":
    """
    Validates input state or injected DataFrame for required columns.
    Returns a properly initialized QuestionState.

    Args:
        state: An existing QuestionState object (if available).
        questions: A list of questions to initialize a new QuestionState if state is None.
        injected_value: A DataFrame to inject into a new QuestionState if state is None.
        state_required_cols: List of columns required in state.insights.
        injected_required_cols: List of columns required in the injected DataFrame.

    Returns:
        QuestionState: A valid state object with all required columns.

    Raises:
        ValueError: If neither state nor injected_value is provided,
                    or if required columns are missing,
                    or if 'paper_id' contains any NA values.
    """
    
    # --- PATH A: Existing State Provided ---
    if state is not None:
        # Strict check: Ensure they didn't try to provide Path B arguments too
        if questions is not None or injected_value is not None:
            raise ValueError("Provide EITHER 'state' OR ('questions' AND 'injected_value'), not both.")

        # Column Validation
        if not set(state_required_cols).issubset(state.insights.columns):
            raise ValueError(f"State.insights missing required columns: {state_required_cols}")
            
        if "paper_id" in state.insights.columns and state.insights["paper_id"].isna().any():
            raise ValueError("State.insights contains NA values in 'paper_id'.")
            
        return state

    # --- PATH B: New State via Injection ---
    elif questions is not None and injected_value is not None:
        if not set(injected_required_cols).issubset(injected_value.columns):
            raise ValueError(f"Injected DataFrame missing: {injected_required_cols}")

        # Fill missing columns
        for field in state_required_cols:
            if field not in injected_value.columns: # Use .columns check directly
                injected_value[field] = np.nan

        return QuestionState(questions=questions, insights=injected_value)

    # --- PATH C: Failure (Nothing provided or partial Path B) ---
    else:
        raise ValueError(
            "Invalid arguments. You must provide a 'state' object "
            "OR both 'questions' and 'injected_value'."
        )
    

def call_chat_completion(llm_client, ai_model, sys_prompt, user_prompt, return_json: bool, fall_back: Dict[str, Any]):
    """
    Call the chat completion API.

    If return_json=True:
        - call with response_format=json_object
        - on error: return fall_back (empty dict)
        - on success: json.loads(...) and return the parsed dict
    If return_json=False:
        - call without response_format
        - on error: return fall_back (empty string)
        - on success: return raw text
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if return_json:
        # -------- JSON MODE --------
        print("Calling LLM for JSON response...")
        try:
            response = llm_client.chat.completions.create(
                model=ai_model,
                messages=messages,
                response_format={"type": "json_object"}, 
                temperature=0
            )
        except Exception as e:
            print(f"Call to OpenAI failed. Error: {e}")
            return fall_back

        # parse JSON response
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

    
    # now = datetime.datetime.now()
    # end_time = now + datetime.timedelta(seconds=timeout + 10)

    # print(
    #     f"Undertaking AI-assisted research. Process will finish by {end_time.strftime('%Y-%m-%d %H:%M:%S')}."
    #     " If not finished by then, the system may have hung."
    # )

    # # Call the LLM
    # try:
    #     response=llm_client.responses.create(
    #         model=ai_model,
    #         input=prompt,
    #         tools=[{"type": "web_search_preview"}],
    #         timeout=timeout
    #     )
    # except llm_client.error.Timeout as e:
    #     print(f"{e} Consider increasing timeout (default is 1200s).")
    #     return None
    # except Exception as e:
    #     print(f"Call to OpenAI failed. Error: {e}")
    #     return None
    # return response.output_text

# def llm_json_clean(x, sys_prompt, llm_client, ai_model, fall_back):

#     response = call_chat_completion(llm_client=llm_client, 
#                                     ai_model=ai_model, 
#                                     sys_prompt=sys_prompt, 
#                                     user_prompt=x, 
#                                     return_json=True, 
#                                     fall_back=fall_back)
    
#     return response

# def json_format_check(x):
#     # Check if its valid json for loading
#     try:
#         response_dict = json.loads(x)
#     except json.JSONDecodeError:
#         error = "The LLM did not return valid json. Efforts to resolve this have failed. Try run the LLM call again."
#         return False, error, x
    
#     result_key = list(response_dict.keys())[0]
#     #Convert the list of dicts to a df
#     response_df = pd.DataFrame(response_dict[result_key]).reset_index(drop = True)
    
#     # Loop to catch any level of escaping (i.e., "strings of strings")
#     while True:
#         # 1. Identify WHICH elements are still strings
#         # This is the reliable check, NOT the column's overall dtype
#         is_string_mask = response_df["paper_author"].apply(lambda val: isinstance(val, str))
        
#         # 2. Stop condition: If NO elements are strings, we are done un-escaping
#         if not is_string_mask.any():
#             break

#         try:
#             # 3. Apply json.loads ONLY to the elements identified as strings
#             response_df.loc[is_string_mask, "paper_author"] = (
#                 response_df.loc[is_string_mask, "paper_author"].apply(json.loads)
#             )
#         except json.JSONDecodeError:
#             # If we fail to un-escape a string element, it's malformed JSON
#             error = "The LLM failed to generate the paper authors as valid json after attempts to repair escaping. Try the LLM call again."
#             return False, error, x
    
#     # --- FINAL VALIDATION CHECK ---
    
#     # Final Check: Ensure every element is definitely a list
#     # The column's dtype will be 'object', so we must check the content
#     if not all(isinstance(val, list) for val in response_df["paper_author"]):
#         error = "The LLM did not return the paper authors as lists. This cannnot be fixed generically. Try call the LLM again"
#         return False, error, x
    
#     # Check whether this output will save to a parquet file
#     response_df_check = response_df.copy()
#     response_df_check["paper_author"] = response_df_check["paper_author"].apply(json.dumps)
#     try:
#         pa.Table.from_pandas(response_df_check)
#     except Exception as e:
#         error = "Despite producing a valid dataframe with authors as lists, the dataframe will fail when saving to parquet. Manually inspect the output to understand the issue"
#         return False, error, x 

#     return True, None, response_df


def restart_pipeline(saves_location = os.path.join(os.getcwd(), "data", "runs")):
    # This function is a placeholder for restarting the pipeline environment.
    # It identifies the latest completed step and provides instructions to continue.
    
    def gen_pipeline_step(latest_file_path):

        #Get the latest path - i.e. excluding _done - to give the path to the state files
        latest_path = os.path.dirname(latest_file_path)
        # Get the latest step name to pass to the pipeline steps dictionary to get the correct text for the user
        latest_step = os.path.basename(latest_path)
        
        pipeline_steps = {
        "01_search_strings": ("You have generated search strings and saved them in your state. "
                              "You should continue with retreieving academic literature. Initialize AcademicLit as follows:\n"
                              f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                              "getlit.AcademicLit(state = latest_state)"),
        "02_academic_lit": ("You have retrieved academic literature and added it to your state. "
                           "You should continue with processing the literature. Initialize AcademicLit as follows:\n"
                           f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                           "getlit.GreyLiterature(state = latest_state, llm_client=llm_client)"),
        "03_grey_lit": ("You have acquired the relevant grey literature and added it to your state. "
                        "You should continue with the next step. Initialize the next class as follows:\n"
                        f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                        "getlit.Literature(state = latest_state)"),
        "04_literature_deduped": ("You have deduplicated the literature and updated your state. "
                                  "You should continue by condutcing an ai assisted check of your literature. Initialize the next class as follows:\n"
                                  f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                                  "getlit.AiLiteratureCheck(state = latest_state, llm_client=llm_client)"),
        "05_ai_lit_check": ("You have completed the AI literature check and updated your state. "
                           "You should proceed to set up your download architecture for your papers. Initialize the next class as follows:\n"
                           f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                           "getlit.DownloadManager(state = latest_state)"),
        "06_full_text_and_chunks": ("You have ingested the full text of your papers, confirmed metadata, and chunked them. You should proceed to generate insights. Initialize the next class as follows:\n"
                                    f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                                    "core.InsightsGenerator(state = latest_state, llm_client=llm_client)"),
        "07_insights": ("You have generated insights from your papers. You should proceed to the next step. Initialize the next class as follows:\n"
                        f"latest_state = state.QuestionState.load(filepath = '{latest_path}')\n"
                        "core.Clustering(state = latest_state, llm_client=llm_client, embedding_model='text-embedding-3-small')")
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

