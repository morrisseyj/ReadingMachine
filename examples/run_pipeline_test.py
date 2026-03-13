"""
ReadingMachine Example Pipeline
--------------------------------

This script demonstrates a full ReadingMachine workflow using a small
toy corpus of academic papers.

The pipeline performs the following stages:

    documents → chunks → insights → clusters → themes → synthesis → report

Before running this script:

1. Download the toy corpus listed in `examples/toy_corpus.md`
2. Place the PDFs in:

        data/docs/

3. Add your OpenAI API key to `.env`

        OPENAI_API_KEY=your_key_here

This script is intentionally verbose and heavily commented so that users
can understand what each step of the ReadingMachine workflow does.
"""

# ==========================================================
# Imports
# ==========================================================

from readingmachine import core, utils, state, render, config

from dotenv import load_dotenv
from openai import OpenAI

import os
import pandas as pd
import random


# ==========================================================
# Load API credentials
# ==========================================================

# Load environment variables from .env
load_dotenv()

# Read OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create LLM client
llm_client = OpenAI(api_key=OPENAI_API_KEY)


# ==========================================================
# Define research questions
# ==========================================================

# These questions define the analytical frame used to extract insights.
# All extracted insights will be associated with one or more of these questions.

questions = [
    "What effects does remote work have on worker productivity?",
    "How does remote work affect worker satisfaction and retention?",
    "What organizational practices enable successful remote work?",
    "What constraints limit the adoption of remote work?",
    "How has the role of remote work changed over time?"
]


# Context helps the LLM interpret the purpose of the corpus
paper_context = (
    "This literature review examines the growth of remote and hybrid work arrangements "
    "and their implications for productivity, worker well-being, and organizational structure. "
    "The goal of the review is to synthesize empirical findings about when remote work "
    "improves or reduces productivity, how it affects job satisfaction and retention, "
    "and what organizational practices enable successful implementation."
)


# ==========================================================
# Prepare initial state objects
# ==========================================================

# ReadingMachine uses a CorpusState object to track the structure
# of the corpus as it moves through the pipeline.

# Create canonical question IDs
questions_dict = {
    f"question_{idx}": q
    for idx, q in enumerate(questions)
}

questions_df = pd.DataFrame(
    list(questions_dict.items()),
    columns=["question_id", "question_text"]
)

# Initialize an insights table with question metadata
# (actual insights will be added later)
insights_df = questions_df.copy()


# ==========================================================
# 1. INGEST DOCUMENTS
# ==========================================================

# The Ingestor loads documents from disk, extracts text, confirms metadata,
# and breaks the text into chunks suitable for LLM reading.

ingestor = core.Ingestor(
    questions=questions_df,
    papers=insights_df,
    llm_client=llm_client,
    ai_model="gpt-4o",
    file_path=os.path.join(os.getcwd(), "data", "docs")
)

# Read PDF/HTML files and extract text
ingestor.ingest_papers()

# Validate metadata against the text (title, author, year)
# Metadata is treated as a first-class object in the pipeline
ingestor.update_metadata()

# Break each document into bounded segments (~paragraph scale)
# This ensures the LLM reads manageable text windows
ingestor.chunk_papers()


# ==========================================================
# Pipeline recovery helper
# ==========================================================

# If a run is interrupted, you can call:
#
# utils.restart_pipeline()
#
# This prints the most recent saved state and tells you how to resume.


# ==========================================================
# 2. GENERATE INSIGHTS
# ==========================================================

# The Insights class performs two passes over the corpus:
#
# 1. Chunk-level insight extraction
# 2. Whole-document meta-insight extraction

insights_generator = core.Insights(
    corpus_state=ingestor.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-4o",
    paper_context=paper_context
)

# Extract claims from each chunk
insights_generator.get_chunk_insights()

# Extract higher-level arguments spanning entire documents
insights_generator.get_meta_insights()


# ==========================================================
# 3. CLUSTER INSIGHTS
# ==========================================================

# Clustering groups semantically similar insights across documents.
# Clusters act as organizational scaffolding for theme generation.

cluster = core.Clustering(
    corpus_state=insights_generator.corpus_state,
    llm_client=llm_client,
    embedding_model="text-embedding-3-small"
)

# Set seed to make clustering reproducible
random.seed(config.seed)

# Convert insights to vector embeddings
cluster.embed_insights()

# ----------------------------------------------------------
# Dimensionality reduction
# ----------------------------------------------------------

# Insight embeddings exist in high-dimensional space.
# We use UMAP to reduce dimensionality before clustering.

# We sweep UMAP parameters and evaluate how well insights
# separate by research question (a proxy for preserved structure).

cluster.tune_umap_params(
    n_neighbors_list=[5, 15, 30, 50, 75, 100],
    min_dist_list=[0.0, 0.1, 0.2, 0.5],
    n_components_list=[5, 10, 20],
    metric_list=["cosine", "euclidean"]
)

# Reduce dimensions using chosen parameters
cluster.reduce_dimensions(
    n_neighbors=75,
    min_dist=0,
    n_components=5,
    metric="cosine",
    random_state=config.seed
)

# ----------------------------------------------------------
# Clustering
# ----------------------------------------------------------

# Tune HDBSCAN clustering parameters
cluster.tune_hdbscan_params(
    min_cluster_sizes=[5, 10, 15, 20],
    metrics=["euclidean", "manhattan"],
    cluster_selection_methods=["eom", "leaf"]
)

# Apply clustering
cluster.generate_clusters({
    "question_0": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "leaf"},
    "question_1": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "leaf"},
    "question_2": {"min_cluster_size": 10, "metric": "euclidean", "cluster_selection_method": "leaf"},
    "question_3": {"min_cluster_size": 5, "metric": "euclidean", "cluster_selection_method": "eom"},
    "question_4": {"min_cluster_size": 10, "metric": "manhattan", "cluster_selection_method": "eom"}
})

# Optional: collapse very small clusters into outliers
cluster.clean_clusters()

# ==========================================================
# 4. SYNTHESIZE THEMES
# ==========================================================

# The Summarize class performs the thematic synthesis pipeline.

summarize = core.Summarize(
    corpus_state=cluster.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-4o",
    paper_output_length=10000
)

# ----------------------------------------------------------
# Cluster summarization
# ----------------------------------------------------------

# Each cluster is summarized individually.
# Clusters are processed in shortest-path order through embedding space
# to improve narrative coherence.

summarize.summarize_clusters()

# ----------------------------------------------------------
# Theme generation
# ----------------------------------------------------------

# Generate an initial theme schema
summarize.gen_theme_schema()

# Map insights to themes
summarize.map_insights_to_themes()

# Populate theme summaries
summarize.populate_themes()

# Identify and reinsert missing insights ("orphans")
summarize.address_orphans()

# ----------------------------------------------------------
# Iterative refinement
# ----------------------------------------------------------

# Regenerate themes incorporating previously orphaned insights
summarize.gen_theme_schema()
summarize.map_insights_to_themes()
summarize.populate_themes()
summarize.address_orphans()

# Reduce redundancy across themes
summarize.address_redundancy()

# ==========================================================
# 5. RENDER FINAL OUTPUT
# ==========================================================

renderer = render.Render(
    summary_state=summarize.summary_state,
    corpus_state=summarize.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-4o"
)

# Optional cosmetic passes

renderer.stylistic_rewrite()
renderer.gen_exec_summary()
renderer.gen_question_summaries()

# Combine render artifacts into a final dataframe
renderer.integrate_cosmetic_changes()

# Export results
renderer.render_output("docx", use_stylized=True)
renderer.render_output("md", use_stylized=True)
renderer.render_output("pdf", use_stylized=True)

print("\nPipeline complete.")