"""
Persistent state management for the ReadingMachine pipeline.

This module defines the two primary state objects used to persist and
reconstruct analytical progress throughout the ReadingMachine workflow:

    CorpusState
    SummaryState

Together these classes separate the reading layer from the synthesis
layer, allowing corpus representation and thematic synthesis to evolve
independently while remaining linked through a common insight-level
representation.

Corpus reading
--------------
Handled by `CorpusState`.

CorpusState stores the structured representations generated during
document ingestion, chunking, and insight extraction. It captures the
progressive transformation of source material into an inspectable
corpus-level representation:

    documents
        ↓
    full_text
        ↓
    chunks
        ↓
    insights

The `insights` table serves as the primary analytical representation of
the corpus. Downstream processes such as embedding generation,
clustering, theme construction, orphan detection, and synthesis operate
on insights rather than directly on source documents.

CorpusState preserves the intermediate structures required for
traceability between synthesized outputs and source material by
maintaining the relationships between documents, chunks, and extracted
insights.

Thematic synthesis
------------------
Handled by `SummaryState`.

SummaryState stores the iterative synthesis artifacts generated from the
insight representation held in CorpusState. Rather than storing a single
final summary, it preserves the sequence of synthesis outputs produced as
the thematic structure is refined.

The synthesis workflow proceeds through successive stages:

    cluster summaries
        ↓
    theme schema generation
        ↓
    insight-to-theme mapping
        ↓
    theme population
        ↓
    orphan detection and reinsertion
        ↓
    theme refinement / re-theming
        ↓
    redundancy handling (optional)

Each stage generates one or more DataFrame artifacts that are appended
to stage-specific lists. This preserves the history of synthesis passes
and enables inspection of how thematic structures evolve over time.

Persistence
-----------
Both state objects support persistence through Parquet serialization.

CorpusState:
    Stores analytical tables as separate Parquet files representing the
    current corpus representation.

SummaryState:
    Stores synthesis artifacts as sequential Parquet files representing
    the history of thematic synthesis.

This architecture supports:

- resumable workflows
- inspection of intermediate artifacts
- iterative synthesis development
- reproducible analytical runs

Fingerprinting
--------------
Both state objects implement deterministic fingerprint functions.

Fingerprints are generated from normalized DataFrame representations and
are used to validate analytical state during resume operations. These
hashes help detect unexpected changes in corpus or synthesis state
between runs.

Design principles
-----------------
The state architecture reflects several methodological principles of
ReadingMachine:

Coverage preservation
    The corpus representation is maintained separately from synthesis
    artifacts so that thematic outputs can always be regenerated from the
    complete insight set.

Inspectability
    Intermediate analytical artifacts are preserved rather than
    discarded, allowing users to examine how representations evolve.

Traceability
    Structured intermediate representations maintain links between
    synthesized outputs and underlying source material.

Reproducibility
    Analytical state can be persisted, restored, fingerprinted, and
    compared across runs.

Separation of reading and synthesis
    Corpus reading and thematic synthesis are represented by distinct
    state objects, mirroring the methodological separation between
    corpus mapping and thematic organization described in the
    ReadingMachine framework.

Together, these state objects provide the persistence layer for
ReadingMachine, enabling large-scale corpus reading and iterative
thematic synthesis to be managed as a structured, inspectable workflow.
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
import pprint
from uuid import uuid4
import time

class CorpusState:
    """
    Persistent state container for ReadingMachine's corpus-reading layer.

    CorpusState stores the structured intermediate representations generated
    during corpus ingestion, chunking, and insight extraction. It serves as
    the primary state object for the reading phase of the ReadingMachine
    pipeline, providing a stable, inspectable representation of the corpus
    prior to embedding, clustering, thematic organization, and synthesis.

    The state is organized into four core tables:

    - **questions**
        Research questions that guide corpus reading and insight extraction.

    - **full_text**
        Full document text indexed by `paper_id`.

    - **chunks**
        Chunked document segments derived from source documents.

    - **insights**
        Extracted insight-level representations of the corpus, including
        atomic claims, arguments, findings, and other research-question-
        relevant statements.

    Together these tables represent successive transformations of source
    documents into a structured analytical representation:

        documents
            ↓
        full_text
            ↓
        chunks
            ↓
        insights

    In ReadingMachine, insights function as the primary unit of corpus
    representation and provide the foundation for downstream processes such
    as embedding generation, clustering, theme construction, orphan
    detection, and synthesis.

    The class provides utilities for:

    - schema validation during initialization
    - persistence to and restoration from Parquet files
    - CSV export and manual-edit workflows
    - normalization of question metadata
    - deterministic state fingerprinting
    - data-format normalization for serialization

    Notes
    -----
    CorpusState is designed to support ReadingMachine's emphasis on
    inspectable intermediate representations. Rather than compressing the
    corpus directly into synthesized outputs, the pipeline maintains
    structured artifacts that can be validated, reviewed, persisted, and
    reused across stages of analysis.

    The `insight_id` field functions as the primary identifier for
    individual insights and provides the linkage between corpus extraction
    and downstream analytical structures.

    While CorpusState is commonly treated as immutable after insight
    generation, the class does not enforce immutability and supports
    controlled modification during workflow steps such as manual review,
    deduplication, and state restoration.
    """
    def __init__(
        self,
        questions: pd.DataFrame,
        insights: pd.DataFrame,
        full_text: Optional[pd.DataFrame] = None,
        chunks: Optional[pd.DataFrame] = None,
    ) -> None:
        
        """
        Initialize a CorpusState object.

        CorpusState stores the shared tabular state used across the ReadingMachine
        pipeline, including research questions, extracted insights, source document
        text, and chunked document segments.

        Parameters
        ----------
        questions : pd.DataFrame
            Research questions guiding the analysis. Must include:
                - `question_id`
                - `question_text`

        insights : pd.DataFrame
            Insight-level analytical data associated with the research questions.
            Must include:
                - `question_id`
                - `question_text`

        full_text : pd.DataFrame, optional
            Full source-document text. If provided, must include:
                - `paper_id`
                - `full_text`
            If omitted, an empty DataFrame with these columns is created.

        chunks : pd.DataFrame, optional
            Chunked source-document text. If provided, must include:
                - `paper_id`
                - `chunk_id`
                - `chunk_text`
            If omitted, an empty DataFrame with these columns is created.

        Raises
        ------
        ValueError
            If any provided DataFrame is missing required columns.

        Notes
        -----
        This state object reflects the ReadingMachine design of maintaining
        inspectable intermediate representations, including questions, insights,
        full text, and chunks, rather than collapsing corpus reading into a single
        synthesis step.
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
        Rebuild the `question_text` column from the unique question mappings
        present in the insights table.

        Some pipeline stages may remove, duplicate, or alter the
        `question_text` column. This method reconstructs the column by
        extracting the unique `(question_id, question_text)` pairs currently
        stored in `self.insights` and merging them back onto the dataset.

        The procedure:

        1. Extract unique `(question_id, question_text)` pairs.
        2. Remove rows with missing `question_text`.
        3. Drop the existing `question_text` column from `self.insights`.
        4. Merge the extracted mapping back onto the table using `question_id`.

        This method assumes that each `question_id` is associated with a
        consistent question text. If multiple question texts exist for the same
        `question_id`, those mappings will be preserved and may result in
        duplicate rows after the merge.

        Modifies
        --------
        self.insights : pd.DataFrame
            Replaces the existing `question_text` column with a version rebuilt
            from the unique question mappings present in the insights table.
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
        Save all CorpusState DataFrame attributes as Parquet files.

        Each attribute on the CorpusState instance is expected to be a
        `pd.DataFrame`. Each DataFrame is written to `save_location` as a
        separate Parquet file named after the attribute, for example
        `questions.parquet`, `insights.parquet`, `full_text.parquet`, and
        `chunks.parquet`.

        Before writing, common missing-value placeholders are normalized to
        `None`. If a `paper_date` column is present, it is coerced to pandas'
        nullable integer dtype. Embedding columns named `full_insight_embedding`
        and `reduced_insight_embedding`, when present, are serialized explicitly
        as Arrow `list<float32>` columns.

        Parameters
        ----------
        save_location : str
            Directory where the Parquet files should be written. The directory is
            created if it does not already exist.

        Raises
        ------
        ValueError
            If any attribute on the CorpusState instance is not a pandas
            DataFrame.

        Notes
        -----
        A `_done` marker file is written after all DataFrame attributes have been
        successfully saved. This can be used by resume logic to identify a
        completed save operation.
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
        Export CorpusState tables to CSV files.

        Writes selected DataFrame attributes to CSV files for inspection,
        debugging, or manual analysis. Prior to export, unnamed columns are
        removed via `_drop_unnamed_columns()`.

        Parameters
        ----------
        save_location : str, default=os.path.join(os.getcwd(), "data", "csv")
            Directory where CSV files will be written. The directory is created
            if it does not already exist.

        write_questions : bool, default=True
            If True, export `self.questions` to `questions.csv`.

        write_insights : bool, default=True
            If True, export `self.insights` to `insights.csv`.

        write_full_text : bool, default=True
            If True, export `self.full_text` to `full_text.csv`.

        write_chunks : bool, default=True
            If True, export `self.chunks` to `chunks.csv`.

        Notes
        -----
        CSV export is intended primarily for human inspection, debugging, and
        external review of intermediate ReadingMachine artifacts. For state
        persistence and pipeline resumption, the Parquet-based `save()` method
        should be used instead, as it preserves data types more reliably.
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

        Reads all `.parquet` files in `filepath`, converts them to pandas
        DataFrames, removes columns whose names begin with `"Unnamed"`, and uses
        files named `questions.parquet`, `insights.parquet`, `full_text.parquet`,
        and `chunks.parquet` to reconstruct a CorpusState instance.

        Parameters
        ----------
        filepath : str
            Path to the directory containing saved Parquet files.

        Returns
        -------
        CorpusState
            A reconstructed CorpusState instance. If `full_text.parquet` or
            `chunks.parquet` is absent, the class initializer creates empty
            DataFrames for those attributes.

        Raises
        ------
        FileNotFoundError
            If `filepath` does not exist.

        FileNotFoundError
            If no `.parquet` files are found in `filepath`.

        ValueError
            If required DataFrames are missing required columns, as enforced by
            the CorpusState initializer.

        Notes
        -----
        This method is intended to load state previously written by `save()`.
        The `_done` marker file is ignored because only files ending in
        `.parquet` are read.

        Parquet files with names other than the expected CorpusState attributes
        are read into the temporary loading dictionary but are not attached to the
        returned CorpusState instance.
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
        Load an insights table from a CSV or Excel file.

        This utility is used when a ReadingMachine insights table has been
        manually edited outside the pipeline (for example, duplicate review,
        paper validation, or corpus curation) and needs to be reloaded into the
        workflow.

        The function loads a CSV or Excel file into a pandas DataFrame and
        optionally validates that a specified set of columns is present. It does
        not reconstruct a CorpusState object, as state restoration requirements
        vary across stages of the pipeline. The returned DataFrame is intended to
        be incorporated into the existing state by downstream workflow-specific
        logic.

        Parameters
        ----------
        filepath : str
            Path to the CSV (`.csv`) or Excel (`.xlsx`, `.xls`) file containing
            insights data.

        encoding : str, default="utf-8"
            Text encoding used when reading CSV files. Ignored for Excel files.

        output_cols : List, default=[]
            Optional list of columns that must be present in the file. When
            provided, only these columns are returned. A ValueError is raised if
            any requested columns are missing.

        Returns
        -------
        pd.DataFrame
            The loaded insights DataFrame. If `output_cols` is specified, the
            returned DataFrame contains only those columns.

        Raises
        ------
        FileNotFoundError
            If `filepath` does not exist.

        ValueError
            If the file extension is not supported or if any requested
            `output_cols` are missing from the loaded file.

        Notes
        -----
        This function performs file loading and optional schema validation only.
        It does not verify that the returned DataFrame conforms to any specific
        ReadingMachine insight schema beyond the columns explicitly requested via
        `output_cols`.
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
        Generate a deterministic fingerprint of the current insight set.

        The fingerprint is computed from a normalized representation of the
        following columns in `self.insights`:

        - `insight_id`
        - `insight`
        - `cluster`

        To ensure stable hashing, the selected data is normalized by:

        1. Replacing missing values with a sentinel string (`"__NULL__"`).
        2. Sorting columns alphabetically.
        3. Sorting rows by the selected columns.
        4. Resetting the row index.
        5. Serializing the result to JSON.

        The resulting JSON representation is hashed using SHA-256.

        Returns
        -------
        str
            SHA-256 hexadecimal digest representing the normalized insight
            dataset.

        Notes
        -----
        This fingerprint is used to verify that a pipeline stage is operating on
        the same underlying insight representation that was used to generate
        previous outputs. It supports resume validation and detection of
        unexpected modifications to the insight set.

        The fingerprint depends only on the `insight_id`, `insight`, and
        `cluster` columns. Changes to other columns in `self.insights` will not
        affect the generated hash.
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

        Constructs a new CorpusState instance using deep copies of the current
        state tables:

        - `questions`
        - `insights`
        - `full_text`
        - `chunks`

        The returned object is fully independent of the original. Changes to any
        DataFrame in the copied state will not affect the corresponding
        DataFrame in the source state.

        Returns
        -------
        CorpusState
            A new CorpusState instance containing deep copies of all state
            DataFrames.

        Notes
        -----
        The copy is created by constructing a new CorpusState through the class
        initializer. As a result, the copied state is subject to the same column
        validation checks performed during normal initialization.
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
        """
        Remove columns whose names begin with 'Unnamed' from all state tables.

        Iterates over all CorpusState attributes and, for each pandas DataFrame,
        drops columns matching the pattern `^Unnamed`. These columns commonly
        arise when CSV files are written with an index and later reloaded.

        Modifies
        --------
        self.questions : pd.DataFrame
            Removes unnamed columns if present.

        self.insights : pd.DataFrame
            Removes unnamed columns if present.

        self.full_text : pd.DataFrame
            Removes unnamed columns if present.

        self.chunks : pd.DataFrame
            Removes unnamed columns if present.

        Notes
        -----
        Only attributes that are pandas DataFrames are processed. All other
        attributes are ignored.
        """
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                clean_value = value.loc[:, ~value.columns.str.contains("^Unnamed")]
                self.__dict__[key] = clean_value
    
    @staticmethod
    def _strict_literal_eval(value):
        """
        Safely parse a string representation of a Python literal.

        This helper is primarily used when loading columns that store Python
        data structures (for example, lists serialized as strings in CSV or
        Excel files). Missing values are converted to empty lists. All other
        values are parsed using `ast.literal_eval()`.

        Parameters
        ----------
        value : Any
            Value to parse. Typically a string representation of a Python
            literal such as a list, dictionary, tuple, number, or string.

        Returns
        -------
        Any
            The parsed Python object. Missing values return an empty list.

        Raises
        ------
        ValueError
            If the value cannot be parsed by `ast.literal_eval()`. The error
            message includes the offending value and guidance on the expected
            format.

        Notes
        -----
        Unlike `eval()`, this method uses `ast.literal_eval()`, which only
        evaluates Python literals and does not execute arbitrary code.

        The current implementation returns an empty list for missing values,
        making it particularly suitable for columns that are expected to contain
        serialized lists.
        """
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
        """
        Convert NumPy array values to Python lists in selected columns.

        Iterates over all DataFrame attributes stored on the CorpusState and,
        for each specified column, converts any values that are NumPy arrays
        into standard Python lists using `tolist()`.

        Parameters
        ----------
        columns : Iterable[str]
            Column names to inspect for NumPy array values.

        Modifies
        --------
        All DataFrame attributes on the CorpusState
            For each specified column that exists in a DataFrame, values of type
            `numpy.ndarray` are replaced with their list representation. All
            other values are left unchanged.

        Notes
        -----
        This method operates on every DataFrame attribute stored in the state,
        not just the canonical `questions`, `insights`, `full_text`, and
        `chunks` tables.

        The conversion is useful when preparing state for serialization,
        export, JSON encoding, or other operations that do not natively support
        NumPy array objects.

        Only values that are instances of `numpy.ndarray` are converted.
        Existing lists and other data types are preserved unchanged.
        """
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
    Persistent state container for ReadingMachine's thematic synthesis layer.

    SummaryState stores the iterative synthesis artifacts generated from the
    insight-level corpus representation held in `CorpusState`. Whereas
    `CorpusState` captures the structured reading of the corpus
    (documents → chunks → insights), `SummaryState` captures the successive
    attempts to organize, synthesize, audit, and refine that representation.

    The synthesis workflow operates over the complete insight set and
    produces a sequence of derived artifacts:

        cluster summaries
            ↓
        theme schema generation
            ↓
        insight-to-theme mapping
            ↓
        theme population
            ↓
        orphan detection and reinsertion
            ↓
        theme refinement / re-theming
            ↓
        redundancy handling (optional)

    Each synthesis stage generates a DataFrame artifact that is appended to
    a stage-specific list. Rather than overwriting prior outputs, the class
    retains the full history of synthesis passes, enabling inspection of how
    the thematic structure evolves across iterations.

    Attributes
    ----------
    cluster_summary_list : List[pd.DataFrame]
        Cluster-level summaries generated from semantically grouped
        insights. Typically contains a single entry.

    theme_schema_list : List[pd.DataFrame]
        Theme schemas generated during each synthesis iteration.

    mapped_theme_list : List[pd.DataFrame]
        Insight-to-theme mappings for each synthesis iteration.

    populated_theme_list : List[pd.DataFrame]
        Theme summaries generated from mapped insight sets.

    orphan_list : List[pd.DataFrame]
        Outputs associated with orphan detection and coverage auditing.

    redundancy_list : List[pd.DataFrame]
        Final redundancy-reduced synthesis outputs.

    Notes
    -----
    SummaryState functions as a persistent synthesis history. The object
    records how the thematic representation of the corpus evolves through
    repeated schema generation, mapping, population, orphan handling, and
    refinement passes.

    This design supports:

    - inspectability of intermediate synthesis artifacts
    - rewind and resume workflows
    - reproducibility of synthesis trajectories
    - debugging and evaluation of thematic evolution
    - auditing of information preservation during synthesis

    Unlike CorpusState, which stores the canonical insight-level
    representation of the corpus, SummaryState stores the lineage of
    artifacts generated while synthesizing that representation.
    """
    def __init__(
        self,
        summary_save_location: str = config.SUMMARY_SAVE_LOCATION,
    ) -> None:
        """
        Initialize an empty SummaryState.

        Creates the summary artifact directory if it does not already exist and
        initializes empty in-memory containers for each summary-stage artifact
        tracked by the class.

        Parameters
        ----------
        summary_save_location : str, default=config.SUMMARY_SAVE_LOCATION
            Directory where summary-state artifacts are persisted.

        Attributes Initialized
        ----------------------
        summary_save_location : str
            Location used for saving and loading summary artifacts.

        cluster_summary_list : list
            Container for cluster-level summaries.

        theme_schema_list : list
            Container for generated theme schemas.

        mapped_theme_list : list
            Container for theme-mapping outputs.

        populated_theme_list : list
            Container for populated theme summaries.

        orphan_list : list
            Container for orphan-detection and orphan-reinsertion outputs.

        redundancy_list : list
            Container for redundancy-pass outputs.
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
        Load a SummaryState from persisted summary artifacts.

        Creates a new SummaryState instance and populates its artifact
        collections by loading previously saved files from the summary-state
        directory. Each artifact type is loaded independently using the
        configured filename prefixes.

        The loaded artifacts represent the history of the thematic synthesis
        process, including theme-schema iterations, theme mappings, theme
        population passes, orphan-detection outputs, and optional redundancy
        processing.

        Parameters
        ----------
        summary_save_location : str, default=config.SUMMARY_SAVE_LOCATION
            Directory containing saved summary-state artifacts.

        Returns
        -------
        SummaryState
            A SummaryState instance populated with all available summary
            artifacts found in the specified directory.

        Notes
        -----
        Artifact loading is delegated to `_load_attribute_from_file()`, which
        retrieves all files matching a given artifact prefix and returns them in
        saved order.

        The resulting state preserves the iterative structure of the
        ReadingMachine synthesis process, including repeated theme-generation,
        theme-mapping, theme-population, and orphan-handling passes.
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
        Load a sequence of persisted synthesis artifacts.

        Searches the summary-state directory for Parquet files whose names begin
        with the specified prefix, loads them as pandas DataFrames, validates the
        resulting sequence, and returns them in chronological order.

        Parameters
        ----------
        file_prefix : str
            Filename prefix identifying a particular synthesis artifact type
            (for example, cluster summaries, theme schemas, or populated
            themes).

        Returns
        -------
        list[pd.DataFrame]
            List of loaded DataFrames ordered from oldest to newest. Returns an
            empty list if no matching files are found.

        Notes
        -----
        Files are discovered using the pattern
        `{file_prefix}*.parquet` within `summary_save_location`.

        Ordering is determined using file creation time (`st_birthtime`) when
        available, falling back to modification time (`st_mtime`) on platforms
        that do not expose creation timestamps.

        After loading, the sequence is validated using
        `_assert_state_integrity()` before being returned.

        This method is used to reconstruct the historical sequence of synthesis
        artifacts generated during iterative theme generation, mapping,
        population, orphan handling, and redundancy reduction.
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
        Perform non-blocking integrity checks on summary-state artifacts.

        This helper inspects a sequence of loaded or generated synthesis
        artifacts and emits warnings when common structural issues are detected.
        The checks are intended for debugging and state diagnostics rather than
        strict validation.

        Current checks include:

        - verifying that the input is a list
        - detecting `None` entries
        - detecting objects that are not DataFrames
        - checking that `theme_id` columns retain an integer dtype

        Parameters
        ----------
        df_list : list
            Sequence of objects expected to contain pandas DataFrames.

        context : str, default=""
            Optional label describing where the check is being performed
            (for example, "Load", "Save", or "Post-populate"). Included in
            warning messages to aid debugging.

        Notes
        -----
        This method intentionally emits warnings rather than raising exceptions.
        SummaryState artifacts represent intermediate synthesis outputs, and
        allowing execution to continue can be useful when diagnosing state drift,
        serialization issues, or schema inconsistencies.

        The primary integrity check currently focuses on `theme_id`, as dtype
        drift in this field can lead to ordering, mapping, and join errors during
        iterative theme synthesis.
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

        Writes each summary-state artifact list to a temporary directory as a
        sequence of Parquet files, then atomically swaps the completed temporary
        directory into `summary_save_location`. Existing saved state is first
        renamed to a backup directory and is only removed after the new directory
        has been successfully moved into place.

        Each artifact list is saved using the configured summary-state prefixes.
        Files are numbered according to their position in the list, for example
        `theme_schema_list_1.parquet`, `theme_schema_list_2.parquet`, and so on.

        Raises
        ------
        Exception
            Re-raises any exception encountered during serialization or directory
            replacement. If possible, the previous saved state is restored before
            the exception is raised.

        Notes
        -----
        Before each artifact list is written, `_assert_state_integrity()` is
        called to emit non-blocking warnings about possible structural issues.

        The directory replacement logic uses retrying renames to handle transient
        Windows file-locking errors. If saving fails, the temporary directory is
        removed and the prior saved state is restored when it has already been
        moved to backup.
        """
        # rename retry util
        def rename_with_retry(src: Path, dst: Path, attempts: int = 60) -> None:
            """
            Rename a path, retrying on transient Windows PermissionError locks.
            """
            last_error = None

            for i in range(attempts):
                try:
                    src.rename(dst)
                    return
                except PermissionError as e:
                    print(f"Rename attempt {i+1}/{attempts} failed due to PermissionError. Retrying...")
                    last_error = e
                    time.sleep(min(0.1 * (i + 1), 2.0))

            raise last_error
    
        # Main function
        save_path = Path(self.summary_save_location)

        #Check parent path exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        #create the temp path and backup path
        temp_path = save_path.parent / f"{save_path.name}_tmp_{uuid4().hex}"
        backup_path = save_path.parent / f"{save_path.name}_old_{uuid4().hex}"

        temp_path.mkdir()

        try:
            for name in config.summary_state_prefix.values():
                data_list = getattr(self, name)

                self._assert_state_integrity(data_list, context=f"Saving: {name}")

                for idx, df in enumerate(data_list, start=1):
                    filename = f"{name}_{idx}.parquet"
                    df.to_parquet(temp_path / filename, index=False)

            if save_path.exists():
                rename_with_retry(save_path, backup_path)

            rename_with_retry(temp_path, save_path)

            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)

        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

            # Restore prior good state if replacement failed after backup rename
            if backup_path.exists() and not save_path.exists():
                rename_with_retry(backup_path, save_path)

            raise

        print(f"Summary outputs saved successfully to {save_path}.")
        

    def rewind_to(self, stage: str, index: int):
        """
        Rewind SummaryState to a specified synthesis stage and iteration.

        Truncates the iterative synthesis artifact lists so that the saved
        summary state reflects progress only through the requested stage and
        iteration. This is useful when rerunning part of the synthesis pipeline
        from an earlier theme-schema, mapping, population, or orphan-handling
        pass.

        Parameters
        ----------
        stage : str
            Stage to retain through. Must be one of:

            - `"schema"`: retain theme schemas through `index`
            - `"mapping"`: retain theme schemas and mapped themes through `index`
            - `"populate"`: retain theme schemas, mappings, and populated themes
            through `index`
            - `"orphan"`: retain theme schemas, mappings, populated themes, and
            orphan outputs through `index`

        index : int
            Zero-based iteration index to retain for the target stage.

        Raises
        ------
        ValueError
            If `stage` is not valid.

        ValueError
            If `index` is negative.

        ValueError
            If `index` exceeds the available entries for the selected stage.

        Notes
        -----
        Rewinding realigns the dependent synthesis artifact lists so that later
        stages do not remain ahead of earlier stages. The redundancy list is
        always cleared, since redundancy reduction is only valid after the final
        synthesis state has been reached.

        After truncating the in-memory artifact lists, this method immediately
        calls `save()` to persist the rewound state to disk.
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
        Reset SummaryState and remove all persisted synthesis artifacts.

        Clears all in-memory synthesis artifacts and deletes the corresponding
        saved summary-state directory. This effectively resets the thematic
        synthesis process to an empty state while leaving the underlying corpus
        representation unchanged.

        Parameters
        ----------
        confirm : str, optional
            Confirmation response. If set to `"yes"` or `"no"`, the method uses
            that value directly. Otherwise, the user is prompted interactively
            until a valid response is provided.

        Behavior
        --------
        If confirmation is `"yes"`:

        - Clears all in-memory synthesis artifact lists.
        - Deletes the summary-state directory and all saved artifacts.
        - Recreates an empty summary-state directory.

        If confirmation is `"no"`:

        - Leaves both in-memory and persisted state unchanged.

        Notes
        -----
        This operation affects only SummaryState and its associated synthesis
        artifacts, including cluster summaries, theme schemas, theme mappings,
        populated themes, orphan outputs, and redundancy outputs.

        CorpusState is not modified. The underlying corpus reading layer,
        including questions, documents, chunks, and insights, remains intact.

        This method is primarily intended for restarting the synthesis process
        from the beginning while reusing an existing corpus representation.

        Warning
        -------
        This is a destructive operation. When confirmed, all persisted synthesis
        history is permanently removed from `summary_save_location`.
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
        Report the current progress of the summary pipeline.

        Inspects the lengths of the SummaryState artifact lists and infers the
        most recently completed synthesis stage. In normal mode, the method
        prints the current artifact counts and a recommended next step. In
        diagnostic mode, it returns a compact machine-readable status dictionary.

        Parameters
        ----------
        diagnostic : bool, default=False
            If False, print the artifact-list counts and a human-readable status
            message. If True, return a dictionary instead of printing the inferred
            status message.

        Returns
        -------
        dict or None
            If `diagnostic=True`, returns a dictionary with:

            - `stage`: inferred most recently completed stage
            - `max_stage`: maximum artifact-list length across tracked stages

            If `diagnostic=False`, prints status information and returns None.

        Notes
        -----
        The method infers progress from these artifact lists:

        - `cluster_summary_list`
        - `theme_schema_list`
        - `mapped_theme_list`
        - `populated_theme_list`
        - `orphan_list`
        - `redundancy_list`

        Stage inference assumes artifacts are generated in the expected
        ReadingMachine synthesis order:

            cluster summaries
                ↓
            theme schemas
                ↓
            mapped themes
                ↓
            populated themes
                ↓
            orphan handling
                ↓
            redundancy handling

        Because theme synthesis can iterate, multiple entries may exist in the
        schema, mapping, population, and orphan lists. The inferred stage is based
        on which list has reached the most recent pass.

        If any redundancy artifact exists, the method treats the summary process
        as complete and reports the `"redundancy"` stage.
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
        Generate a deterministic fingerprint of the current SummaryState.

        The fingerprint is computed from the most recent DataFrame in each
        summary artifact list:

        - `cluster_summary_list`
        - `theme_schema_list`
        - `mapped_theme_list`
        - `populated_theme_list`
        - `orphan_list`
        - `redundancy_list`

        For each non-empty list, the latest DataFrame is normalized, serialized
        to JSON, and hashed with SHA-256. Empty lists contribute the sentinel
        string `"EMPTY"`. The resulting per-list hashes are concatenated and
        hashed again to produce the final fingerprint.

        Returns
        -------
        str
            SHA-256 hexadecimal digest representing the current summary state.

        Notes
        -----
        Only the most recent artifact in each list contributes to the
        fingerprint. Earlier iterations in the synthesis history do not affect
        the hash unless they are also the latest artifact for their list.

        DataFrames are normalized before hashing by replacing missing values,
        inferring object dtypes, sorting columns, sorting rows by all columns,
        and resetting the index.

        This fingerprint supports resume validation and detection of unexpected
        changes to the current synthesis state.
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

        Constructs a new SummaryState instance and populates it with deep copies
        of all synthesis artifact lists:

        - `cluster_summary_list`
        - `theme_schema_list`
        - `mapped_theme_list`
        - `populated_theme_list`
        - `orphan_list`
        - `redundancy_list`

        Each DataFrame in each artifact list is copied independently, producing
        a fully detached summary state that can be modified without affecting the
        original.

        Returns
        -------
        SummaryState
            A new SummaryState instance containing deep copies of all synthesis
            artifacts.

        Notes
        -----
        The copied state retains the same `summary_save_location` as the source
        state but does not write any files to disk. Persistence occurs only if
        `save()` is subsequently called.

        Unlike CorpusState, SummaryState preserves the full synthesis history.
        The copied object therefore contains copies of all recorded iterations
        of theme generation, mapping, population, orphan handling, and
        redundancy processing.
        """
        new_state = SummaryState(summary_save_location=self.summary_save_location)
        new_state.cluster_summary_list = [df.copy(deep=True) for df in self.cluster_summary_list]
        new_state.theme_schema_list = [df.copy(deep=True) for df in self.theme_schema_list]
        new_state.mapped_theme_list = [df.copy(deep=True) for df in self.mapped_theme_list]
        new_state.populated_theme_list = [df.copy(deep=True) for df in self.populated_theme_list]
        new_state.orphan_list = [df.copy(deep=True) for df in self.orphan_list]
        new_state.redundancy_list = [df.copy(deep=True) for df in self.redundancy_list]
        return new_state




