
# Import custom libraries
from lit_review_machine import utils
from lit_review_machine import config

# Import standard libraries
import pandas as pd
from typing import Optional, List
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import ast
from pathlib import Path
import shutil
import hashlib
import json
import pprint


class CorpusState:
    """
    Container for managing research pipeline state.

    This class keeps track of:
      1. Questions dictionary - key values of question id and question text.
      2. Insights dataframe - traces insights back to the `paper_id`.
      3. Chunk dataframe - links text chunks to a `paper_id` and `chunk_id`.
      4. Full-text dataframe - links full text to a `paper_id`.

    It provides methods to save and load the entire state object as a pickle,
    and to initialize a state from a CSV containing literature data.
    """

    def __init__(
        self,
        questions: pd.DataFrame,
        insights: pd.DataFrame,
        full_text: Optional[pd.DataFrame] = None, 
        chunks: Optional[pd.DataFrame] = None
    ) -> None:
        
        self.questions = questions
        
        required_insights_cols = ["question_id", "question_text"]
        if not all(col in insights.columns for col in required_insights_cols):
            raise ValueError(
                "insights dataframe requires the following variables to initialize: 'question_id' and 'question_text'."
            )
        self.insights = insights
        
        
        if full_text is not None:
            required_full_text_cols = ["paper_id", "full_text"]
            if not all(col in full_text.columns for col in required_full_text_cols):
                raise ValueError(
                    "full_text dataframe must include 'paper_id' and 'full_text'."
                )
            self.full_text = full_text
        else:
            self.full_text = pd.DataFrame(columns=["paper_id", "full_text"])
        
        if chunks is not None:
            required_chunks_cols = ["paper_id", "chunk_id", "chunk_text"]
            if not all(col in chunks.columns for col in required_chunks_cols):
                raise ValueError(
                    "chunks dataframe must include 'paper_id', 'chunk_id', 'chunk_text'."
                )
            self.chunks = chunks
        else:
            self.chunks = pd.DataFrame(columns=["paper_id", "chunk_id", "chunk_text"])

    def enforce_canonical_question_text(self) -> None:
        """
        Ensures that state.insights always uses the canonical question_text for each question_id.
        """
        # Build canonical mapping
        canonical = (
            self.insights[["question_id", "question_text"]]
            .drop_duplicates()
            .dropna(subset=["question_text"])
        )
        # Drop any possibly incorrect question_text
        self.insights = self.insights.drop(columns=["question_text"], errors="ignore")
        # Merge canonical question_text back in
        self.insights = self.insights.merge(canonical, on="question_id", how="left")

    # ---------------------------------------------------------------------- #
    #                            SAVE / EXPORT                              #
    # ---------------------------------------------------------------------- #

    
    def save(self, save_location: str) -> None:
        """
        Save the entire CorpusState object as Parquet files (one per DataFrame attribute).
        Handles list-like columns (`paper_author`, embeddings) using PyArrow array types.
        """

        os.makedirs(save_location, exist_ok=True)

        for key, value in self.__dict__.items():
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"Attribute {key} must be a pandas DataFrame.")

            df_to_save = value.copy()
            df_to_save = df_to_save.replace(["", "NA", pd.NA, np.nan, "null"], None)
            if "paper_date" in df_to_save.columns:
                df_to_save["paper_date"] = pd.to_numeric(df_to_save["paper_date"], errors="coerce").astype("Int64")
            table = pa.Table.from_pandas(df_to_save)

            # for col in ["paper_author", "full_insight_embedding", "reduced_insight_embedding"]:
            for col in ["full_insight_embedding", "reduced_insight_embedding"]:
                if col not in df_to_save.columns:
                    continue

                idx = table.column_names.index(col)
                
                # # ---- list of strings ----
                # if col == "paper_author":
                #     # First make sure NA values empty lists or lists that have [NA] are all converted to None
                #     df_to_save[col] = df_to_save[col].apply(lambda x: x if isinstance(x, list) and len(x) > 0 and not pd.isna(x[0]) else None)
                #     # each cell is a list[str]; Arrow handles NULLs automatically
                #     arr = pa.array(df_to_save[col].tolist(), type=pa.list_(pa.string()))
                #     table = table.set_column(idx, col, arr)

                # ---- embeddings ----
                #else:
                # First make sure NA values or empty lists are all converted to None
                df_to_save[col] = df_to_save[col].apply(lambda x: x if isinstance(x, list) and len(x) > 0 and not pd.isna(x[0]) else None)
                arr = pa.array(
                    [None if x is None else np.asarray(x, np.float32)
                    for x in df_to_save[col]],
                    type=pa.list_(pa.float32())
                )
                table = table.set_column(idx, col, arr)

            # save parquet
            print(f"Saving {key} to Parquet...")
            pq.write_table(table, os.path.join(save_location, f"{key}.parquet"), compression="zstd")

        # marker file
        with open(os.path.join(save_location, "_done"), "w") as f:
            pass

    def write_to_csv(
        self, 
        save_location: str = os.path.join(os.getcwd(), "data", "csv"), 
        write_questions=True,
        write_insights=True, 
        write_full_text=True, 
        write_chunks=True
    ) -> None:
        os.makedirs(save_location, exist_ok=True)
        self._drop_unnamed_columns()
        if write_questions:
            self.questions.to_csv(os.path.join(save_location, "questions.csv"), index=False)
        if write_insights:
            self.insights.to_csv(os.path.join(save_location, "insights.csv"), index=False)
        if write_full_text:
            self.full_text.to_csv(os.path.join(save_location, "full_text.csv"), index=False)
        if write_chunks:
            self.chunks.to_csv(os.path.join(save_location, "chunks.csv"), index=False)

    # ---------------------------------------------------------------------- #
    #                             LOAD METHODS                              #
    # ---------------------------------------------------------------------- #
    
    @classmethod
    def from_json(cls, filepath: str = os.path.join(os.getcwd(), "data", "parquet", "state"), join_str="-||-|||-||-") -> "CorpusState":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")

        files = os.listdir(filepath)
        files = [file for file in files if Path(file).suffix.lower() == ".json"]
        if "insights.json" not in files:
            raise FileNotFoundError(f"'insights.json' file not found in {filepath}, cannot load CorpusState.")
        state_df_dict = {}
        for file in files:
            full_path = os.path.join(filepath, file)
            df = pd.read_json(full_path, orient="records", lines=True)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")] #Remove unnamed cols

            for col in ["paper_author", "insight", "chunks", "pages"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.split(join_str) if pd.notna(x) and x != "" else []
                    )

            state_df_dict[Path(file).stem] = df

        question_state = cls(
            questions=state_df_dict.get("questions"),
            insights=state_df_dict["insights"],
            full_text=state_df_dict.get("full_text", None),
            chunks=state_df_dict.get("chunks", None)
        )

        return question_state

    @classmethod
    def from_parquet(cls, filepath: str = os.path.join(os.getcwd(), "data", "parquet", "state"), new = True, join_str="-||-|||-||-") -> "CorpusState":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")
         
        files = [file for file in os.listdir(filepath) if Path(file).suffix.lower() == ".parquet"]
        state_df_dict = {}
        if new:
            if "insights.parquet" not in files:
                raise FileNotFoundError(f"'insights.parquet' file not found in {filepath}, cannot load CorpusState.")

            for file in files:
                full_path = os.path.join(filepath, file)
                df = pd.read_parquet(full_path)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")] #Remove unnamed cols
                state_df_dict[Path(file).stem] = df
                for col in ["paper_author", "insight", "chunks", "pages"]:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: x.split(join_str) if pd.notna(x) and x != "" else []
                        )

            corpus_state = cls(
                questions=state_df_dict.get("questions"),
                insights=state_df_dict["insights"],
                full_text=state_df_dict.get("full_text", None),
                chunks=state_df_dict.get("chunks", None)
            )

            return corpus_state
       
        else:
            files = os.listdir(filepath)
            state_df_dict = {}
            for file in files:
                full_path = os.path.join(filepath, file)
                df = pd.read_parquet(full_path)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")] #Remove unnamed cols
                state_df_dict[Path(file).stem] = df

            corpus_state = cls(
                questions=state_df_dict.get("questions"),
                insights=state_df_dict.get("insights", None),
                full_text=state_df_dict.get("full_text", None),
                chunks=state_df_dict.get("chunks", None)
            )
            
            # Normalize the columns - first get arrays to lists then get 
            corpus_state.arrays_to_lists(["paper_author", "insight", "chunks", "pages"])
            corpus_state.normalize_list_columns(["paper_author", "insight", "chunks", "pages"])
            return corpus_state
    
    @classmethod
    def load(cls, filepath: str) -> "CorpusState":
        """
        Load a CorpusState object from a folder of Parquet files.
        Expects one Parquet per DataFrame attribute.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")

        state_df_dict = {}

        for file in os.listdir(filepath):
            if not file.endswith(".parquet"):
                continue  # skip _done or other files

            full_path = os.path.join(filepath, file)
            table = pq.read_table(full_path)

            # --- FIX 1: remove stray arg ---
            # to_pandas() takes no filepath argument
            df = table.to_pandas()

            # --- optional cleanup ---
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            state_df_dict[Path(file).stem] = df

        # --- FIX 2: handle missing keys safely ---
        return cls(
            questions=state_df_dict.get("questions"),
            insights=state_df_dict.get("insights"),
            full_text=state_df_dict.get("full_text"),
            chunks=state_df_dict.get("chunks"),
        )

    @classmethod
    def from_csv(cls, filepath: str, encoding="utf-8") -> "CorpusState":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")

        df_dict = {}
        for file in os.listdir(filepath):
            # Check the file is a csv, otherwise skip it
            if Path(file).suffix.lower() != ".csv":
                continue

            full_path = os.path.join(filepath, file)
            df = pd.read_csv(full_path, encoding=encoding)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            df_dict[Path(file).stem] = df

        corpus_state = cls(
            questions=df_dict.get("questions"),
            insights=df_dict.get("insights", None),
            full_text=df_dict.get("full_text", None),
            chunks=df_dict.get("chunks", None)
        )
        
        # Normalize the columns - first get arrays to lists then get viable lists
        return corpus_state

    def fingerprint(self) -> str:
        """
        Deterministic fingerprint of the corpus state.
        Used for resume validation.
        Only includes fields relevant to downstream summarization.
        """

        relevant_columns = ["insight_id", "insight", "cluster"]

        df = self.insights[relevant_columns].copy()

        df_normalized = (
            df.fillna("__NULL__")
            .sort_index(axis=1)
            .sort_values(by=list(df.columns))
            .reset_index(drop=True)
        )

        json_bytes = df_normalized.to_json().encode("utf-8")
        return hashlib.sha256(json_bytes).hexdigest()

    # ---------------------------------------------------------------------- #
    #                             HELPER UTILS                              #
    # ---------------------------------------------------------------------- #
    def _drop_unnamed_columns(self) -> None:
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                clean_value = value.loc[:, ~value.columns.str.contains("^Unnamed")]
                self.__dict__[key] = clean_value
    
    @staticmethod
    def _strict_literal_eval(value):
        if pd.isna(value):
            return []
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError) as e:
            raise ValueError(
                f"Fatal Error: Failed to evaluate literal in 'paper_author' column. "
                f"Ensure ALL entries are strictly formatted as a Python list of strings, "
                f"e.g., ['Author A', 'Author B']. The offending value was: '{value}'. "
                f"Original error: {e.__class__.__name__}"
            ) from e

    def arrays_to_lists(self, columns):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                for col in columns:
                    if col in attr_value.columns:
                        attr_value[col] = attr_value[col].apply(
                            lambda v: v.tolist() if isinstance(v, np.ndarray) else v
                        )
                setattr(self, attr_name, attr_value)

    # def normalize_list_columns(self, columns):
    #     for attr_name, attr_value in self.__dict__.items():
    #         if isinstance(attr_value, pd.DataFrame):
    #             for col in columns:
    #                 if col in attr_value.columns:
    #                     attr_value[col] = attr_value[col].apply(utils.ensure_list_of_strings)
    #             setattr(self, attr_name, attr_value)
    

class SummaryState:
    """
    Holds all interpretive artifacts generated during the summarization stage.

    Attributes
    ----------
    cluster_summary_list : List[pd.DataFrame]
    theme_schema_list : List[pd.DataFrame]
    mapped_theme_list : List[pd.DataFrame]
    populated_theme_list : List[pd.DataFrame]
    orphan_list : List[pd.DataFrame]
    redundancy_list : List[pd.DataFrame]

    This object represents the full state of the thematic summarization pipeline.

    The SummaryState is distinguished from CorpusState as the latter is focused on tracking the way insights are developed, SummaryState tracks the evolution of the summary artefacts
    """
    def __init__(
        self,
        summary_save_location: str = config.SUMMARY_SAVE_LOCATION,
    ) -> None:
        self.summary_save_location = summary_save_location

        # Ensure summary save location exists
        if not os.path.exists(self.summary_save_location):
            os.makedirs(self.summary_save_location, exist_ok=True)

        self.cluster_summary_list = []
        self.theme_schema_list = []
        self.mapped_theme_list = []
        self.populated_theme_list = []
        self.orphan_list = []
        self.redundancy_list = []

    @classmethod
    def load(cls, summary_save_location:str = config.SUMMARY_SAVE_LOCATION) -> "SummaryState":
        """
        Class method that allows us to load summary state from file, which otherwise would initalize with an empty list
        """

        state = cls(summary_save_location=summary_save_location)

        state.cluster_summary_list = state._load_attribute_from_file(config.summary_state_prefix["cluster_summary_list"]) # List of len(1) containing the cluster summaries
        state.theme_schema_list = state._load_attribute_from_file(config.summary_state_prefix["theme_schema_list"]) # List of len(n) containing the theme schemas for each re-theming pass
        state.mapped_theme_list = state._load_attribute_from_file(config.summary_state_prefix["mapped_theme_list"]) # List of len(n) containing the mapped themes for each theme mapping pass
        state.populated_theme_list = state._load_attribute_from_file(config.summary_state_prefix["populated_theme_list"]) # List of len(n) containing the populated themes for each theme population pass
        state.orphan_list = state._load_attribute_from_file(config.summary_state_prefix["orphan_list"]) # List of len(n) containing the orphaned insights for each theme population pass
        state.redundancy_list = state._load_attribute_from_file(config.summary_state_prefix["redundancy_list"]) # List of len(1) containing the output of the final redundancy pass

        return state


    def _load_attribute_from_file(self, file_prefix: str) -> Optional[List[pd.DataFrame]]:
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
            return []

        # 2. Sort by creation/birth time
        # Uses st_birthtime (Creation) if available, falls back to st_mtime (Modified)
        paths_sorted = sorted(
            paths, 
            key=lambda p: getattr(p.stat(), 'st_birthtime', p.stat().st_mtime)
        )

        # 3. Load and return the list of DataFrames
        output = [pd.read_parquet(p.absolute()) for p in paths_sorted]
        self._assert_state_integrity(output, context = f"Load: {file_prefix}")
        return output
    
    def _assert_state_integrity(self, df_list: list, context: str = ""):
        """
        Developer-facing integrity check for summarize pipeline corpus_state.

        - Ensures structural keys like `theme_id` maintain expected dtype.
        - Prints loud warnings but does NOT raise errors.
        - Intended as a guardrail during development and refactoring.

        Args:
            df_list: A list of pandas DataFrames to validate.
            context: Optional string indicating where this check is being run
                    (e.g., 'reload', 'save', 'post-populate').
        """

        if not isinstance(df_list, list):
            print(
                f"⚠️ STATE WARNING [{context}]: Expected list of DataFrames, "
                f"received {type(df_list)}"
            )
            return

        for idx, df in enumerate(df_list):

            if df is None:
                print(
                    f"STATE WARNING [{context}]: "
                    f"DataFrame at index {idx} is None."
                )
                continue

            if not hasattr(df, "columns"):
                print(
                    f"STATE WARNING [{context}]: "
                    f"Object at index {idx} is not a DataFrame "
                    f"(type={type(df)})."
                )
                continue

            # Check structural identity key
            if "theme_id" in df.columns:
                if not pd.api.types.is_integer_dtype(df["theme_id"]):
                    print(
                        f"\nSTATE WARNING [{context}]\n"
                        f"DataFrame index: {idx}\n"
                        f"'theme_id' dtype is {df['theme_id'].dtype}, "
                        f"expected integer.\n"
                        f"This may cause ordering or join errors downstream.\n"
                        f"Correct dtype drift before proceeding.\n"
                )

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

    
    def save(self) -> None:
        """
        Atomically saves the current SummaryState to disk.
        Writes to a temporary directory first, then replaces the live directory.
        """

        save_path = Path(self.summary_save_location)

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a true temporary directory next to target
        temp_path = save_path.parent / f"{save_path.name}_tmp"

        # If temp exists from previous crash, clean it
        if temp_path.exists():
            shutil.rmtree(temp_path)

        temp_path.mkdir()

        # Write all summary state lists into temp directory
        for name in config.summary_state_prefix.values():
            data_list = getattr(self, name)
            
            # Check the data integrity to id any issues. 
            self._assert_state_integrity(data_list, context=f"Saving: {name}")

            for idx, df in enumerate(data_list, start=1):
                filename = f"{name}_{idx}.parquet"
                df.to_parquet(temp_path / filename, index=False)


        # Now atomically replace old directory
        if save_path.exists():
            shutil.rmtree(save_path)

        temp_path.rename(save_path)

        print(f"Summary outputs saved successfully to {save_path}.")
        

    def rewind_to(self, stage: str, index: int):

        stage_order = {
            "schema": 0,
            "mapping": 1,
            "populate": 2,
            "orphan": 3,
        }

        if stage not in stage_order:
            raise ValueError("Invalid stage name.")

        target_depth = stage_order[stage]

        structural = [
            self.theme_schema_list,
            self.mapped_theme_list,
            self.populated_theme_list,
            self.orphan_list,
        ]

        if index < 0:
            raise ValueError("Index must be >= 0.")

        target_list = structural[target_depth]
        if index >= len(target_list):
            raise ValueError("Index exceeds available passes.")

        # Now realign explicitly
        if target_depth == 0:  # schema
            self.theme_schema_list = self.theme_schema_list[:index+1]
            self.mapped_theme_list = self.mapped_theme_list[:index]
            self.populated_theme_list = self.populated_theme_list[:index]
            self.orphan_list = self.orphan_list[:index]

        elif target_depth == 1:  # mapping
            self.theme_schema_list = self.theme_schema_list[:index+1]
            self.mapped_theme_list = self.mapped_theme_list[:index+1]
            self.populated_theme_list = self.populated_theme_list[:index]
            self.orphan_list = self.orphan_list[:index]

        elif target_depth == 2:  # populate
            self.theme_schema_list = self.theme_schema_list[:index+1]
            self.mapped_theme_list = self.mapped_theme_list[:index+1]
            self.populated_theme_list = self.populated_theme_list[:index+1]
            self.orphan_list = self.orphan_list[:index]

        elif target_depth == 3:  # orphan
            self.theme_schema_list = self.theme_schema_list[:index+1]
            self.mapped_theme_list = self.mapped_theme_list[:index+1]
            self.populated_theme_list = self.populated_theme_list[:index+1]
            self.orphan_list = self.orphan_list[:index+1]

        # Always clear redundancy
        self.redundancy_list = []
        
        # Save the state after rewinding (handles the deleting of old files and writes the current state objects to file)
        self.save()

    def restart(self, confirm = None):
        while confirm not in ["yes", "no"]:
            confirm = input(
                "Are you sure you want to restart? This will clear all summary state. Enter 'yes' or 'no':\n"
            ).lower()

        if confirm == "yes":
            # Clear in-memory state
            self.cluster_summary_list = []
            self.theme_schema_list = []
            self.mapped_theme_list = []
            self.populated_theme_list = []
            self.orphan_list = []
            self.redundancy_list = []

            # Clear disk state
            save_path = Path(self.summary_save_location)
            if save_path.exists():
                shutil.rmtree(save_path)

            save_path.mkdir(parents=True, exist_ok=True)

            print("SummaryState reset successfully.")
        else:
            print("Restart cancelled.")

    def status(self, diagnostic = False):
        # Get the status of the current summary state by checking the lengths of each list of artifacts.            
        status = {
            "cluster_summary_list": len(self.cluster_summary_list),
            "theme_schema_list": len(self.theme_schema_list),
            "mapped_theme_list": len(self.mapped_theme_list),
            "populated_theme_list": len(self.populated_theme_list),
            "orphan_list": len(self.orphan_list),
            "redundancy_list": len(self.redundancy_list)
        }
        if not diagnostic:
            # Pretty print the status with indentation for readability
            pprint.pprint(status, indent=4)
        
        # Calculate  maximum number of passes completed across all stages
        max_stage = max([val for val in status.values()])
        # Now determine where in the process the user must be based on which objects have equal the latest run
        if max_stage == 0:
            if diagnostic:
                return {"stage": "no_runs", "max_stage": max_stage}
            else:
                print("No summarization runs detected. You can start the summarization process by running the cluster summary stage.")
                return None
        if status["redundancy_list"] > 0:
            if diagnostic:
                return {"stage": "redundancy", "max_stage": max_stage}
            else:
                print("You Summaries are complete. You can now proceed to instantiate the render class")
                return None
        if status["orphan_list"] == max_stage:
            if diagnostic:
                return {"stage": "orphan", "max_stage": max_stage}
            else:
                print(f"You have identified orphaned insights on your most recent run (run: {max_stage}). Your next step should be to handle redundancy.")
                return None
        elif status["populated_theme_list"] == max_stage:
            if diagnostic:
                return {"stage": "populated_theme", "max_stage": max_stage}
            else:
                print(f"You have populated themes on your most recent run (run: {max_stage}). Your next step should be to handle handle orphans.")
                return None
        elif status["mapped_theme_list"] == max_stage:
            if diagnostic:
                return {"stage": "mapped_theme", "max_stage": max_stage}
            else:
                print(f"You have mapped themes on your most recent run (run: {max_stage}). Your next step should be to populate themes.")
                return None
        elif status["theme_schema_list"] == max_stage:
            if diagnostic:
                return {"stage": "theme_schema", "max_stage": max_stage}
            else:
                print(f"You have generated a theme schema on your most recent run (run: {max_stage}). Your next step should be to map themes.")
                return None
        else:
            if diagnostic:
                return {"stage": "cluster_summary", "max_stage": max_stage}
            else:
                print(f"You have generated a cluster summary on your most recent run (run: {max_stage}). Your next step should be to generate a theme schema.")
                return None

    def fingerprint(self):
        """
        Deterministic fingerprint of the summary state.
        Uses the latest DataFrame from each summary artifact list.
        If anything changes, resume becomes invalid.
        """

        parts = []

        state_lists = [
            self.cluster_summary_list,
            self.theme_schema_list,
            self.mapped_theme_list,
            self.populated_theme_list,
            self.orphan_list,
            self.redundancy_list,
        ]

        for df_list in state_lists:
            if df_list:
                df = df_list[-1].copy()

                df = (
                    df.fillna("__NULL__")
                    .sort_index(axis=1)
                    .sort_values(by=sorted(df.columns))
                    .reset_index(drop=True)
                )

                json_bytes = df.to_json().encode("utf-8")
                hashed_df = hashlib.sha256(json_bytes).hexdigest()

                parts.append(hashed_df)
            else:
                parts.append("EMPTY")

        combined = "".join(parts).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()




