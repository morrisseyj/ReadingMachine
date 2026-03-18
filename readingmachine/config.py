"""
Configuration settings for ReadingMachine.

This module centralizes filesystem paths, pipeline constants, and
naming conventions used throughout the ReadingMachine codebase.
Keeping these settings in a single location ensures that all pipeline
components reference consistent directories and file naming schemes.

The configuration primarily defines:

    • pipeline state storage locations
    • summary artifact locations
    • rendering artifact locations
    • output directories
    • serialization naming conventions
    • reproducibility parameters

All paths are resolved relative to the current working directory
unless explicitly overridden.

Design principle
----------------

The configuration module intentionally avoids runtime logic and
contains only static parameters so that pipeline behavior remains
predictable and reproducible across runs.
"""

import os

###
# FILESYSTEM PATHS
###
"""
Filesystem paths used by the ReadingMachine pipeline.

These directories store intermediate state objects, rendering artifacts,
and final outputs generated during pipeline execution.
"""

# Location where users place thier document corpus
CORPUS_LOCATION = os.path.join(os.getcwd(), "data", "corpus")

# Location in which the object states are located and which are updated afgter each step of the pipeline completes
STATE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "runs")

# Location in which the summary objects are saved after the summarization step of the pipeline completes
SUMMARY_SAVE_LOCATION = os.path.join(STATE_SAVE_LOCATION, "summaries")

# Location in which the render objects are saved after the rendering step of the pipeline completes
RENDER_SAVE_LOCATION = os.path.join(STATE_SAVE_LOCATION, "renders")

# Location in which the final outputs are saved after the output generation from the render class
OUTPUT_SAVE_LOCATION = os.path.join(os.getcwd(), "outputs")

# Location in which the paper chunks are saved after the chunking step of the pipeline completes
PICKLE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "pickles")

FUZZY_CHECK_PATH = os.path.join(os.getcwd(), "data", "fuzzy_checks")

###
# REPRODUCIBILITY CONFIGURATION
###
"""
Random seed used to ensure reproducibility across pipeline runs.
"""

# The seed value to ensure reproducibulity across runs
seed = 42

###
# SUMMARY STATE PREFIXES
###
"""
Filename prefixes used when serializing SummaryState artifacts.

Each synthesis stage writes a sequence of DataFrames to disk. These
prefixes ensure consistent naming across pipeline runs.
"""

# Filename prefixes for summary objects:
summary_state_prefix = {
    "cluster_summary_list": "cluster_summary_list",
    "theme_schema_list": "theme_schema_list",
    "mapped_theme_list": "mapped_theme_list",
    "populated_theme_list": "populated_theme_list",
    "orphan_list": "orphan_list",
    "redundancy_list": "redundancy_list"
}

###
# RENDER ARTIFACT CONFIGURATION
###
"""
Filename used to store the deterministic hash of the summary DataFrame.

This hash ensures that rendering artifacts correspond to the correct
analytical state and prevents accidental reuse of artifacts generated
from different synthesis outputs.
"""

# Filename for render objects:
summary_hash = "summary_hash.parquet" # The hash for the summary df passed on init

"""
Filename prefixes used for persisted render artifacts.

These artifacts include cosmetic transformations applied during the
render stage, such as stylistic rewrites and executive summaries.
"""

render_prefix = {
    "final_render_df": "final_render_df", # The final render df that is populated with the artifacts from the render process and is used to generate the final output
    "render_title_exec_summary_df": "render_title_exec_summary_df", # The df that contains the title and executive summary for the render
    "render_question_summary_df": "render_question_summary_df", # The df that contains the question summaries for the render
    "render_stylized_rewrite_df": "render_stylized_rewrite_df" # The df that contains the stylized rewrites for the render
}

