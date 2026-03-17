
import importlib

def reload():
    from readingmachine.tools import getlit
    from readingmachine import utils, config

    importlib.reload(config)
    importlib.reload(utils)
    importlib.reload(getlit)

    return(None)

#----------------------
"""
ReadingMachine Example: Corpus Discovery (getlit)

This example demonstrates how to build a literature corpus using the
optional `getlit` tools before running the core ReadingMachine pipeline.

Workflow:

    research questions
        → search strings
        → academic literature retrieval
        → grey literature retrieval
        → duplicate detection
        → AI literature completeness check
        → download architecture

The output of this script is a populated `CorpusState` containing
candidate documents. Once the papers are downloaded, the corpus can
be passed to the core ReadingMachine pipeline.

Environment setup:

    1. Install dependencies via `uv sync`
    2. Set your OpenAI API key in `.env`
    3. (Optional) set EMAIL_DOMAIN for polite Crossref/OpenAlex API usage
"""

# ==========================================================
# Imports
# ==========================================================

from readingmachine.tools import getlit
from readingmachine import utils

from dotenv import load_dotenv
from openai import OpenAI

import os


# ==========================================================
# Load API credentials
# ==========================================================

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
llm_client = OpenAI(api_key=OPEN_API_KEY)


# ==========================================================
# Define research questions
# ==========================================================

questions = [
    "What effects does remote work have on worker productivity?",
    "How does remote work affect worker satisfaction and retention?",
    "What organizational practices enable successful remote work?",
    "What constraints limit the adoption of remote work?",
    "How has the role of remote work changed over time?"
]


paper_context = (
    "This literature review examines the growth of remote and hybrid work arrangements "
    "and their implications for productivity, worker well-being, and organizational structure. "
    "The goal of the review is to synthesize empirical findings about when remote work "
    "improves or reduces productivity, how it affects job satisfaction and retention, "
    "and what organizational practices enable successful implementation."
)


# ==========================================================
# Optional: restart helper
# ==========================================================

# If a previous run was interrupted you can call:
#
# utils.restart_pipeline()
#
# This prints the most recent saved pipeline step and shows
# how to resume from that state.

utils.restart_pipeline()


# ==========================================================
# 1. Generate search strings
# ==========================================================

# The ScholarSearchString class converts research questions
# into structured academic search queries.

search_strings = getlit.ScholarSearchString(
    questions=questions,
    llm_client=llm_client,
    num_prompts=2   # number of search prompts per question
)

search_strings.searchstring_maker()


# ==========================================================
# 2. Retrieve academic literature
# ==========================================================

# AcademicLit queries academic APIs such as Crossref and OpenAlex.

academic_lit = getlit.AcademicLit(
    corpus_state=search_strings.corpus_state
)

#------------

# Each search string returns `num_results` papers.
# Total results ≈ prompts × results × questions x 2 search engines (currently 2 search engines) (before deduplication).

academic_lit.search_crossref(num_results=5)
academic_lit.search_openalex(num_results=5)


# ==========================================================
# 3. Retrieve grey literature
# ==========================================================

grey_literature = getlit.GreyLiterature(
    corpus_state=academic_lit.corpus_state,
    llm_client=llm_client
)

# Provide example organizations for the reasoning model
example_grey_literature_sources = (
    "Gallup, Stanford Institute for Economic Policy Research (SIEPR), "
    "The Conference Board, International Labour Organization (ILO)"
)

grey_literature.get_grey_lit(
    example_grey_literature_sources=example_grey_literature_sources
)


# ==========================================================
# 4. Remove duplicates
# ==========================================================

literature = getlit.Literature(
    corpus_state=grey_literature.corpus_state
)

# Remove exact duplicates automatically
literature.drop_exact_duplicates()

# Export fuzzy matches for manual review
literature.get_fuzzy_matches()

# ----------------------------------------------------------
# Manual Step
# ----------------------------------------------------------
#
# Review the exported CSV files located in:
#
#    data/fuzzy_check/
#
# Remove any duplicate rows manually.
#
# After editing the CSV files run:

literature.update_state()


# ==========================================================
# 5. AI literature completeness check
# ==========================================================

# This step uses an LLM to identify potentially missing
# academic or grey literature.

ai_literature = getlit.AiLiteratureCheck(
    corpus_state=literature.corpus_state,
    llm_client=llm_client
)

ai_literature.ai_literature_check()


# ==========================================================
# 6. Create download architecture
# ==========================================================

############
# Set up the system for file downloads
############

# Instantiate the downloader
downloads = getlit.DownloadManager(
    corpus_state=ai_literature.corpus_state
)

# ----------------------------------------------------------
# MANUAL STEP
# ----------------------------------------------------------
#
# The DownloadManager creates a folder structure under:
#
#     data/corpus/
#
# Each research question receives its own folder.
#
# You should manually download the identified papers and place them
# in the appropriate question folders.
#
# Ideally, the downloaded files should use the paper_id as the filename
# (for example: paper_id.pdf). This is not strictly required, but it
# greatly improves traceability between downloaded files and the
# literature metadata stored in the CorpusState.
#
# Once the files have been downloaded, update the state using:

downloads.update()


# ----------------------------------------------------------
# NEXT STEP: RUN THE CORE READINGMACHINE PIPELINE
# ----------------------------------------------------------
#
# At this point you have constructed a corpus containing:
#
#     - research questions
#     - academic literature
#     - grey literature
#     - deduplicated results
#     - AI literature suggestions
#
# The updated corpus_state can now be passed to the core
# ReadingMachine ingestion stage.
#
# Example initialization:
#
#     from readingmachine import core
#
#     ingestor = core.Ingestor(
#         corpus_state = downloads.corpus_state,
#         llm_client = llm_client,
#         ai_model = "gpt-4o"
#     )
#
# The full reading pipeline is demonstrated in:
#
#     examples/run_core_pipeline.py
#
# That script performs the structured reading workflow:
#
#     documents
#         → chunking
#         → insight extraction
#         → clustering
#         → thematic synthesis
#         → report generation
#