# Config file for lit_review_machine

import os

# Location in which the object states are located and which are updated afgter each step of the pipeline completes
STATE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "runs")

# Location in which the summary objects are saved after the summarization step of the pipeline completes
SUMMARY_SAVE_LOCATION = os.path.join(STATE_SAVE_LOCATION, "summaries")

# Location in which the render objects are saved after the rendering step of the pipeline completes
RENDER_SAVE_LOCATION = os.path.join(STATE_SAVE_LOCATION, "renders")

# Location in which the paper chunks are saved after the chunking step of the pipeline completes
PICKLE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "pickles")

# Filename prefixes for summary objects:
summary_state_prefix = {
    "cluster_summary_list": "cluster_summary_list",
    "theme_schema_list": "theme_schema_list",
    "mapped_theme_list": "mapped_theme_list",
    "populated_theme_list": "populated_theme_list",
    "orphan_list": "orphan_list",
    "redundancy_list": "redundancy_list"
}

# Filename for render objects:
summary_hash = "summary_hash.parquet" # The hash for the summary df passed on init

render_prefix = {
    "final_render_df": "final_render_df", # The final render df that is populated with the artifacts from the render process and is used to generate the final output
    "render_title_exec_summary_df": "render_title_exec_summary_df", # The df that contains the title and executive summary for the render
    "render_question_summary_df": "render_question_summary_df", # The df that contains the question summaries for the render
    "render_stylized_rewrite_df": "render_stylized_rewrite_df" # The df that contains the stylized rewrites for the render
}