"""
State management for the ReadingMachine analytical pipeline.

This module defines the two persistent state objects that track the
transformation of a document corpus through the ReadingMachine workflow:

    CorpusState
    SummaryState

The objects correspond to two distinct phases of the methodology.

Corpus processing
-----------------
Handled by `CorpusState`.

This stage represents the structured reading of the corpus and records
how raw documents are transformed into analytical units.

The transformation sequence is:

    documents
    → full text
    → chunks
    → insights

The `insights` table acts as the central semantic index of the corpus.
All downstream analysis operates on these extracted claims rather than
directly on the document text.

`CorpusState` therefore preserves the lineage required for traceability:

    theme
    → insight
    → chunk
    → document

Each insight retains identifiers linking it to the originating text
segment, enabling citation-anchored synthesis and auditability.

The corpus representation is intended to be **append-only**. Once
documents are ingested and insights are generated, the corpus state
should be treated as immutable for the duration of an analysis run.
Changes to the corpus (for example adding documents or modifying
extraction prompts) should result in the creation of a new
`CorpusState`.

Thematic synthesis
------------------
Handled by `SummaryState`.

This stage represents the interpretive organization of extracted
insights into thematic structures.

Unlike `CorpusState`, which stores tidy analytical tables,
`SummaryState` records **pipeline passes**. Each stage of the
summarization workflow produces a DataFrame that is appended to a list,
preserving the full history of synthesis iterations.

The synthesis workflow proceeds through the following stages:

    cluster summaries
    → theme schema generation
    → insight-to-theme mapping
    → theme population
    → orphan detection
    → iteration
    → redundancy handling

Each iteration of this process produces a new entry in the relevant
artifact lists. This design allows researchers to inspect how the
thematic structure evolves across synthesis passes.

Persistence
-----------
Both state objects are designed to be persisted to disk using Parquet
serialization.

CorpusState:
    Each analytical table is stored as a separate Parquet file
    (questions, full_text, chunks, insights).

SummaryState:
    Each synthesis pass is stored as an individual Parquet file
    within the summary output directory.

This approach enables:

- resumable pipelines
- inspection of intermediate artifacts
- reproducible analytical runs

Fingerprinting
--------------
Both state objects implement deterministic fingerprint functions.

These hashes are computed from normalized DataFrame representations and
are used to detect changes in the analytical state when resuming a
pipeline. If the fingerprint differs from the expected value, the
pipeline can warn the user that the analytical lineage has changed.

Design principles
-----------------
The state architecture is designed around several core principles:

Traceability
    Every synthesized claim can be traced back to the text segment from
    which it originated.

Inspectability
    Intermediate artifacts are preserved rather than overwritten,
    allowing researchers to examine how analytical structures emerge.

Reproducibility
    Analytical configurations and intermediate states can be persisted
    and restored without altering results.

Separation of phases
    Corpus reading and thematic synthesis are represented by separate
    state objects to preserve conceptual clarity and avoid accidental
    coupling of extraction and interpretation stages.

Together, these state objects provide the backbone of the ReadingMachine
pipeline, allowing large-scale machine reading to be organized into a
structured and inspectable analytical workflow.
"""


# Import custom libraries
from . import config

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
    Persistent representation of the corpus layer of the ReadingMachine pipeline.

    `CorpusState` stores the structured analytical representation of a document
    corpus after ingestion and insight extraction. It acts as the central data
    container for all corpus-level artifacts produced during the reading stage
    of the pipeline.

    The object maintains four primary tables:

    - **questions**: research questions guiding insight extraction
    - **full_text**: full document text indexed by `paper_id`
    - **chunks**: segmented text units derived from documents
    - **insights**: atomic claims extracted from the corpus

    These tables represent successive transformations of the corpus:

        documents
        → full_text
        → chunks
        → insights

    Downstream analytical stages (embedding, clustering, theme synthesis)
    operate primarily on the `insights` table.

    The class provides utilities for:

    - validating required schema elements
    - persisting state to disk
    - restoring state from multiple serialization formats
    - enforcing canonical question text across insights
    - computing deterministic fingerprints for resume validation

    Notes
    -----
    `CorpusState` is designed to be **append-only**. Once insights are generated,
    the corpus representation should be treated as immutable for the remainder
    of an analysis run. Modifying the corpus (for example by adding documents or
    regenerating insights) should instead produce a new `CorpusState`.

    The `insight_id` column in the `insights` table acts as the primary key
    linking corpus extraction to all downstream analytical structures.
    """

    def __init__(
        self,
        questions: pd.DataFrame,
        insights: pd.DataFrame,
        full_text: Optional[pd.DataFrame] = None,
        chunks: Optional[pd.DataFrame] = None,
    ) -> None:
        
        """
        Initialize a new CorpusState.

        Parameters
        ----------
        questions : pd.DataFrame
            DataFrame containing the research questions guiding the analysis.
            Must include `question_id` and `question_text`.

        insights : pd.DataFrame
            Core analytical dataset containing extracted insights.
            Must include at minimum:
                - `question_id`
                - `question_text`

        full_text : pd.DataFrame, optional
            DataFrame containing full document text indexed by `paper_id`.
            If omitted, an empty DataFrame with the expected schema is created.

        chunks : pd.DataFrame, optional
            DataFrame containing chunked text segments derived from the corpus.
            Must include:
                - `paper_id`
                - `chunk_id`
                - `chunk_text`
            If omitted, an empty DataFrame with the expected schema is created.

        Raises
        ------
        ValueError
            If required columns are missing from the provided DataFrames.
        """
        
        required_question_cols = ["question_id", "question_text"]
        if not all(col in questions.columns for col in required_question_cols):
            raise ValueError(
                "questions dataframe requires the following variables to initialize: 'question_id' and 'question_text'."
            )
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
        Enforce a single canonical `question_text` for each `question_id`.

        Insight extraction stages may occasionally introduce duplicated or
        inconsistent representations of the research question text. This
        method reconstructs a canonical mapping between `question_id` and
        `question_text` based on unique values present in the insights table.

        The procedure:

        1. Extract unique `(question_id, question_text)` pairs.
        2. Drop any existing `question_text` column in the insights table.
        3. Re-merge the canonical mapping back into the table.

        This ensures downstream synthesis stages operate on a stable
        representation of the research questions.

        Modifies
        --------
        self.insights
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
        Persist the CorpusState to a directory of Parquet files.

        Each DataFrame attribute of the object is serialized to an individual
        Parquet file. The filenames correspond to the attribute names:

        - questions.parquet
        - insights.parquet
        - full_text.parquet
        - chunks.parquet

        List-like columns and embedding arrays are converted to Arrow list
        types to ensure correct serialization and restoration.

        Parameters
        ----------
        save_location : str
            Directory where the Parquet files should be written.

        Notes
        -----
        A marker file named `_done` is written after all tables have been
        successfully saved. This file can be used by resume logic to detect
        completed state persistence.

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
        """
        Export corpus state tables to CSV files.

        Parameters
        ----------
        save_location : str
            Output directory for the CSV files.

        write_questions : bool
            Whether to export the questions table.

        write_insights : bool
            Whether to export the insights table.

        write_full_text : bool
            Whether to export the full text table.

        write_chunks : bool
            Whether to export the chunk table.

        Notes
        -----
        This export format is primarily intended for manual inspection
        and debugging. Parquet persistence should be used for reliable
        pipeline resumes.
        """

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

    @classmethod 
    def load(cls, filepath: str) -> "CorpusState":
        """
        Load a CorpusState from a directory of Parquet files.

        This method reconstructs a `CorpusState` by reading a directory
        containing one Parquet file per table. Each file is expected to be
        named after the corresponding attribute:

        - questions.parquet
        - insights.parquet
        - full_text.parquet (optional)
        - chunks.parquet (optional)

        The method iterates over all `.parquet` files in the directory,
        converts them to pandas DataFrames, removes any extraneous
        "Unnamed" columns, and maps them to their respective attributes
        based on filename.

        Parameters
        ----------
        filepath : str
            Path to the directory containing Parquet files.

        Returns
        -------
        CorpusState
            A reconstructed `CorpusState` instance populated with the
            available tables. Missing optional tables (`full_text`,
            `chunks`) will be set to empty DataFrames via the class
            initializer.

        Raises
        ------
        FileNotFoundError
            If the specified directory does not exist.

        FileNotFoundError
            If no Parquet files are found in the directory.

        ValueError
            If required tables (`questions`, `insights`) are missing
            required columns, as enforced by the class initializer.

        Notes
        -----
        This method assumes that the directory was created using the
        `CorpusState.save()` method and follows the same naming conventions.

        The presence of additional Parquet files in the directory will be
        ignored unless their filenames match expected attribute names.

        Any `_done` marker file is ignored during loading.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No folder found at {filepath}")
        
        state_df_dict = {}

        parquet_files = [f for f in os.listdir(filepath) if f.endswith(".parquet")]

        if len(parquet_files) == 0:
            raise FileNotFoundError(f"No Parquet files found in {filepath}. Ensure the directory contains the expected Parquet files for questions, insights, full_text, and chunks.")
        
        print(f"Loading from Parquet files in {filepath}...")
        for file in parquet_files:
            full_path = os.path.join(filepath, file)
            table = pq.read_table(full_path)
            df = table.to_pandas()
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            state_df_dict[Path(file).stem] = df

        corpus_state = cls(
            questions=state_df_dict.get("questions"),
            insights=state_df_dict.get("insights", None),
            full_text=state_df_dict.get("full_text", None),
            chunks=state_df_dict.get("chunks", None)
        )

        return(corpus_state)
    


    
    def load_insights_from_csv_xslx(
            filepath: str, 
            encoding: str = "utf-8",
            output_cols: List = []
        ) -> pd.DataFrame:
        """
        Load insights from a CSV or Excel file, ensuring consistent schema.

        Function is used as a general drop in for when you have to grab the csv or xlsx version of the state insights df 
        after manual review of duplicates or after downloading papers.

        This method does not recompose the full state as at different point in the pipeline the process for recomposing after manual edit is different 
        For example after dedup state is rebuilt simply by updating corpus_state.insights with the cleaned insights, 
        whereas during ingestion we have to align corpus_state.full_text and corpus_state.insights so that only the valid paper_ids remain in the state.

        Parameters
        ----------
        filepath : str
            Path to the CSV or Excel file containing insights data. 
        
        encoding : str, default="utf-8"
            Encoding to use when reading a CSV file.
        
        output_cols : List, default=[]
            Optional list of columns to retain in the output DataFrame. If empty, all columns are retained.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the insights data with consistent schema.
        """
        # Check if file exists
        if not os.path.isfile(filepath):
                raise FileNotFoundError(f"No file found at {filepath}. Ensure the insights file is correctly named and located in the specified directory.")
        
        # Load the file based on extension
        if filepath.endswith(".csv"):
            print(f"Loading insights from CSV file at {filepath}...")
            insights_df = pd.read_csv(filepath, encoding=encoding)
        elif filepath.endswith((".xlsx", ".xls")):
            print(f"Loading insights from Excel file at {filepath}...")
            insights_df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        # Ensure the insights matches the expected schmema if passed in output_cols
        if output_cols:
            missing_cols = [col for col in output_cols if col not in insights_df.columns]
            if missing_cols:
                raise ValueError(f"The following specified output columns are missing from the file: {missing_cols}")
            else:
                insights_df = insights_df[output_cols]

        return insights_df


    def fingerprint(self) -> str:
        """
        Generate a deterministic fingerprint of the corpus state.

        The fingerprint is computed from a normalized representation
        of the `insights` table using the following fields:

        - `insight_id`
        - `insight`
        - `cluster`

        This hash is used for pipeline resume validation to ensure
        that downstream synthesis stages are operating on the same
        corpus representation.

        Returns
        -------
        str
            SHA-256 hash representing the normalized insight dataset.
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

    
    def copy(self) -> "CorpusState":
        """
        Create a deep copy of the CorpusState.

        This method creates a new instance of `CorpusState` with deep copies
        of all DataFrame attributes. This is useful for creating modified
        versions of the state without altering the original.

        Returns
        -------
        CorpusState
            A new `CorpusState` instance with copied data.
        """
        return CorpusState(
            questions=self.questions.copy(deep=True),
            insights=self.insights.copy(deep=True),
            full_text=self.full_text.copy(deep=True),
            chunks=self.chunks.copy(deep=True)
        )
    

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


class SummaryState:
    """
    Persistent state container for the thematic synthesis stage of ReadingMachine.

    `SummaryState` stores all interpretive artifacts produced after the corpus
    reading phase. While `CorpusState` represents the structured semantic index
    of the corpus (documents → chunks → insights), `SummaryState` represents the
    **iterative analytical synthesis** of those insights.

    The summarization pipeline operates as a sequence of iterative passes:

        cluster summaries
        → theme schema generation
        → insight-to-theme mapping
        → theme population
        → orphan detection
        → iteration

    Each stage produces a DataFrame artifact that is appended to a list.
    These lists preserve the full history of synthesis passes, allowing
    inspection of how themes evolve across iterations.

    Attributes
    ----------
    cluster_summary_list : List[pd.DataFrame]
        Sequential cluster summaries derived from clustered insights.
        Typically length 1.

    theme_schema_list : List[pd.DataFrame]
        Theme schemas generated during each iteration of thematic synthesis.

    mapped_theme_list : List[pd.DataFrame]
        Insight-to-theme mapping tables for each synthesis pass.

    populated_theme_list : List[pd.DataFrame]
        The synthesized textual summaries for each theme.

    orphan_list : List[pd.DataFrame]
        Insights that were not incorporated into theme summaries and
        require reinsertion.

    redundancy_list : List[pd.DataFrame]
        Final redundancy-corrected theme summaries produced after the
        synthesis process is complete.

    Notes
    -----
    The object functions as a **pipeline history log**. Rather than
    overwriting earlier synthesis artifacts, each stage appends new
    results to the relevant list.

    This design supports:

    - reproducibility
    - inspectability
    - rewind and resume workflows
    - debugging of theme evolution

    Unlike `CorpusState`, which stores tidy analytical datasets,
    `SummaryState` stores **pipeline passes**.
    """
    def __init__(
        self,
        summary_save_location: str = config.SUMMARY_SAVE_LOCATION,
    ) -> None:
        """
        Initialize an empty SummaryState.

        Parameters
        ----------
        summary_save_location : str
            Directory where summary artifacts will be persisted as
            Parquet files. Defaults to the location specified in
            the project configuration.
        """
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
        Load an existing SummaryState from disk.

        This method reconstructs the SummaryState by locating Parquet files
        corresponding to each synthesis artifact and loading them into the
        appropriate lists.

        Files are loaded in chronological order to preserve the historical
        sequence of synthesis passes.

        Parameters
        ----------
        summary_save_location : str
            Directory containing previously saved summary artifacts.

        Returns
        -------
        SummaryState
            Reconstructed summary state containing all previously saved
            synthesis artifacts.
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
        Load a list of DataFrames corresponding to a synthesis artifact.

        This method searches the summary save directory for Parquet files
        matching the given prefix and loads them into a list ordered by
        file creation time.

        Parameters
        ----------
        file_prefix : str
            Filename prefix used to identify the artifact group.

        Returns
        -------
        List[pd.DataFrame] or []
            List of loaded DataFrames ordered from oldest to newest.
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
        Developer-facing integrity checks for summary artifacts.

        The method verifies that loaded or generated synthesis artifacts
        maintain expected structural properties.

        Checks include:

        - confirming objects are DataFrames
        - verifying that `theme_id` columns retain integer dtype
        - warning if unexpected structures are encountered

        The function intentionally **prints warnings instead of raising
        exceptions** so that developers can continue execution while
        diagnosing state inconsistencies.

        Parameters
        ----------
        df_list : list
            List of DataFrames to validate.

        context : str
            Optional label indicating where the check is occurring
            (e.g., "Load", "Save", "Post-populate").
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

    
    def save(self) -> None:
        """
        Persist the current SummaryState to disk.

        All synthesis artifacts are written as Parquet files inside the
        configured summary directory.

        The save procedure is atomic:

        1. Write all files to a temporary directory
        2. Delete the previous directory
        3. Rename the temporary directory to the target location

        This prevents partially written state from corrupting the pipeline
        if execution is interrupted.

        Notes
        -----
        Each synthesis pass is stored as a separate file, allowing
        full reconstruction of the pipeline history.
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
        """
        Rewind the synthesis pipeline to a previous stage.

        This method truncates summary artifact lists so that the pipeline
        appears as though it had only progressed to a specified stage and
        iteration.

        Parameters
        ----------
        stage : str
            Target stage to rewind to. Must be one of:

            - "schema"
            - "mapping"
            - "populate"
            - "orphan"

        index : int
            Iteration index to retain.

        Notes
        -----
        Rewinding automatically clears the redundancy stage and saves
        the updated state to disk.
        """

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
        """
        Reset the entire SummaryState and remove all persisted synthesis artifacts.

        This method clears both the in-memory state and the corresponding files
        stored on disk in the summary save directory. It is intended for situations
        where a user wishes to completely restart the thematic synthesis stage
        from the beginning.

        The method prompts the user for confirmation before proceeding in order
        to prevent accidental deletion of summary artifacts.

        Parameters
        ----------
        confirm : str, optional
            Confirmation flag used to bypass the interactive prompt. If set to
            `"yes"` or `"no"`, the method will use that value directly instead
            of prompting the user.

        Behavior
        --------
        If confirmation is `"yes"`:

        - All in-memory summary artifact lists are cleared.
        - The summary output directory is deleted.
        - A new empty directory is created at the same location.

        If confirmation is `"no"`:

        - No changes are made.

        Notes
        -----
        This operation affects only the summarization stage (`SummaryState`).
        It does not modify the underlying `CorpusState` or any corpus-level
        artifacts such as documents, chunks, or insights.

        This method is primarily intended for interactive CLI workflows where
        users may wish to discard previous synthesis passes and rerun the
        summarization pipeline.
        """
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
        """
        Report the current progress of the summarization pipeline.

        This method inspects the lengths of each synthesis artifact list to
        determine which stage of the summarization workflow has most recently
        completed.

        The pipeline stages are inferred from the following artifact lists:

        - cluster_summary_list
        - theme_schema_list
        - mapped_theme_list
        - populated_theme_list
        - orphan_list
        - redundancy_list

        Parameters
        ----------
        diagnostic : bool, default False
            If False (default), the method prints a human-readable status
            message describing the current pipeline stage and recommended
            next step.

            If True, the method returns a structured dictionary describing
            the current stage and iteration depth instead of printing output.

        Returns
        -------
        dict or None
            If `diagnostic=True`, returns a dictionary containing:

            - `stage` : the inferred pipeline stage
            - `max_stage` : the highest iteration index across artifact lists

            If `diagnostic=False`, the function prints status information
            and returns None.

        Notes
        -----
        The stage inference logic assumes that synthesis artifacts are
        generated sequentially according to the pipeline order:

            cluster summaries
            → theme schema
            → insight mapping
            → theme population
            → orphan detection
            → redundancy handling

        Because synthesis is iterative, multiple entries may exist in
        each artifact list. The function determines the current stage
        by identifying which artifact list contains the most recent
        completed pass.

        This method is primarily intended as a user-facing diagnostic
        tool for interactive workflows.
        """


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
         Generate a deterministic fingerprint of the summarization state.

        The fingerprint is computed using the most recent DataFrame
        from each synthesis artifact list.

        Returns
        -------
        str
            SHA-256 hash representing the current summarization state.

        Notes
        -----
        This hash is used to validate resume operations and detect
        changes to synthesis artifacts between runs.
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
                .infer_objects(copy=False)
                .sort_index(axis=1)
                .sort_values(by=sorted(df.columns), kind="mergesort")
                .reset_index(drop=True)
            )

                json_bytes = df.to_json().encode("utf-8")
                hashed_df = hashlib.sha256(json_bytes).hexdigest()

                parts.append(hashed_df)
            else:
                parts.append("EMPTY")

        combined = "".join(parts).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()
    
    def copy(self) -> "SummaryState":
        """
        Create a deep copy of the SummaryState.

        This method creates a new instance of `SummaryState` with deep copies
        of all synthesis artifact lists. This is useful for creating modified
        versions of the state without altering the original.

        Returns
        -------
        SummaryState
            A new `SummaryState` instance with copied data.
        """
        new_state = SummaryState(summary_save_location=self.summary_save_location)
        new_state.cluster_summary_list = [df.copy(deep=True) for df in self.cluster_summary_list]
        new_state.theme_schema_list = [df.copy(deep=True) for df in self.theme_schema_list]
        new_state.mapped_theme_list = [df.copy(deep=True) for df in self.mapped_theme_list]
        new_state.populated_theme_list = [df.copy(deep=True) for df in self.populated_theme_list]
        new_state.orphan_list = [df.copy(deep=True) for df in self.orphan_list]
        new_state.redundancy_list = [df.copy(deep=True) for df in self.redundancy_list]
        return new_state




