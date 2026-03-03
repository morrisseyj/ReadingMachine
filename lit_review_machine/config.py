# Config file for lit_review_machine

import os

# Location in which the object states are located and which are updated afgter each step of the pipeline completes
STATE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "runs")

# Location in which the summary object is saved after the summarization step of the pipeline completes
SUMMARY_SAVE_LOCATION = os.path.join(STATE_SAVE_LOCATION, "summaries")

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

