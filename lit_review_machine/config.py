# Config file for lit_review_machine

import os

# Location in which the object states are located and which are updated afgter each step of the pipeline completes
STATE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "runs")

# Location in which the summary object is saved after the summarization step of the pipeline completes
SUMMARY_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "summaries")

# Location in which the paper chunks are saved after the chunking step of the pipeline completes
PICKLE_SAVE_LOCATION = os.path.join(os.getcwd(), "data", "pickles")