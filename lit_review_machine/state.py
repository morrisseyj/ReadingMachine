
# Import custom libraries
from lit_review_machine import utils

# Import standard libraries
import pandas as pd
from typing import Optional, List
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import ast
from pathlib import Path

class QuestionState:
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
        Save the entire QuestionState object as Parquet files (one per DataFrame attribute).
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
    def from_json(cls, filepath: str = os.path.join(os.getcwd(), "data", "parquet", "state"), join_str="-||-|||-||-") -> "QuestionState":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")

        files = os.listdir(filepath)
        files = [file for file in files if Path(file).suffix.lower() == ".json"]
        if "insights.json" not in files:
            raise FileNotFoundError(f"'insights.json' file not found in {filepath}, cannot load QuestionState.")
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
    def from_parquet(cls, filepath: str = os.path.join(os.getcwd(), "data", "parquet", "state"), new = True, join_str="-||-|||-||-") -> "QuestionState":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")
         
        files = [file for file in os.listdir(filepath) if Path(file).suffix.lower() == ".parquet"]
        state_df_dict = {}
        if new:
            if "insights.parquet" not in files:
                raise FileNotFoundError(f"'insights.parquet' file not found in {filepath}, cannot load QuestionState.")

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

            question_state = cls(
                questions=state_df_dict.get("questions"),
                insights=state_df_dict["insights"],
                full_text=state_df_dict.get("full_text", None),
                chunks=state_df_dict.get("chunks", None)
            )

            return question_state
       
        else:
            files = os.listdir(filepath)
            state_df_dict = {}
            for file in files:
                full_path = os.path.join(filepath, file)
                df = pd.read_parquet(full_path)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")] #Remove unnamed cols
                state_df_dict[Path(file).stem] = df

            question_state = cls(
                questions=state_df_dict.get("questions"),
                insights=state_df_dict.get("insights", None),
                full_text=state_df_dict.get("full_text", None),
                chunks=state_df_dict.get("chunks", None)
            )
            
            # Normalize the columns - first get arrays to lists then get 
            question_state.arrays_to_lists(["paper_author", "insight", "chunks", "pages"])
            question_state.normalize_list_columns(["paper_author", "insight", "chunks", "pages"])
            return question_state
    
    @classmethod
    def load(cls, filepath: str) -> "QuestionState":
        """
        Load a QuestionState object from a folder of Parquet files.
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
    def from_csv(cls, filepath: str, encoding="utf-8") -> "QuestionState":
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

        question_state = cls(
            questions=df_dict.get("questions"),
            insights=df_dict.get("insights", None),
            full_text=df_dict.get("full_text", None),
            chunks=df_dict.get("chunks", None)
        )
        
        # Normalize the columns - first get arrays to lists then get viable lists
        return question_state

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

    def normalize_list_columns(self, columns):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                for col in columns:
                    if col in attr_value.columns:
                        attr_value[col] = attr_value[col].apply(utils.ensure_list_of_strings)
                setattr(self, attr_name, attr_value)
     