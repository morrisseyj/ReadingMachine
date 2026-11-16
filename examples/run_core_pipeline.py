%load_ext autoreload
%autoreload 2

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

        data/corpus/

3. Add your OpenAI API key to `.env`

        OPENAI_API_KEY=your_key_here

This script is intentionally verbose and heavily commented so that users
can understand what each step of the ReadingMachine workflow does.
"""

# ==========================================================
# Imports
# ==========================================================



from dotenv import load_dotenv
from openai import OpenAI

import os
import pandas as pd
from readingmachine import core, render, config


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
    "What underlying structural features (the conceptual framings, and the institutional arrangements through which they are operationalized) of "
    "the US foreign assistance system limited or undermined poverty reduction, reduced suffering, accountable governance, gender equality, "
    "or local leadership, and how did they shape or relate to the distribution of decision-making authority, control over resources, and "
    "agenda-setting power in ways that contributed to these outcomes?",

    "What underlying structural features (the conceptual framings, and the institutional arrangements through which they are operationalized) of "
    "the US foreign assistance system enabled or strengthened efforts at poverty reduction, reduced suffering, accountable governance, "
    "gender equality, or local leadership, and how did they shape or relate to the distribution of decision-making authority, control over "
    "resources, and agenda-setting power in ways that supported these outcomes?",

    "How were have the dominant structural features (conceptual framings, and operational and institutional forms) been embedded in, transmitted "
    "through, or transformed by wider aid institutions, and with what consequences for the distribution of decision-making authority, control "
    "over resources, and agenda-setting power, as well as for system outcomes?",

    "How and why has political economy of the US foreign assistance system changed over the following overlapping historical periods:\n "
     "1) post world war II reconstruction, development and decolonization\n "
     "2) Cold war developmentism\n "
     "3) Structural adjustment and the debt crisis\n "
     "4) post 1990 liberal internationalism and governance\n "
     "5) post 9/11 securitization\n "
     "6) post 2015 sustainable development\n\n"
     "Political economy should be understood broadly to include a wide range of political concerns and economic interests such that the motivations for the structure of the aid system are the object of analysis.\n\n"
]


# Context helps the LLM interpret the purpose of the corpus
paper_context = (
    "This review is motivated by a desire to understand the structural features of US foreign assistance that have either "
    "enabled or undermined the goals of the foreign assistance system. The intention behind the review is to inform the "
    "structural architecture of a future foreign assistance system by identifying and advancing those structures that "
    "most effectively supported the goals of foreign assistance, while avoiding those that undermined them.\n\n"

    "We recognize that the goals of foreign assistance have shifted over time, and this review is not concerned with "
    "all goals historically pursued by the aid system. Rather, it focuses on goals that remain normatively important "
    "today: poverty reduction, reduced human suffering, accountable governance, gender equality, and local leadership. "
    "Other goals or imperatives — such as securitization, geopolitical competition, market expansion, or stabilization — are "
    "relevant where they shaped, constrained, or undermined pursuit of those normative goals that remain important today.\n\n"

    "This review understands the US foreign assistance system as an institutional architecture composed of underlying conceptual "
    "framings and assumptions about development and humanitarian action. These framings structure and shape the design and "
    "operation of institutional forms—including organizational practices, governance structures, accountability systems, "
    "financing arrangements, knowledge systems, and partnership models—through which they are enacted in practice.  "
    "These institutional forms shape the distribution of decision-making authority, control over resources, and agenda-setting "
    "power within the system. The analysis therefore focuses on how structural features give rise to these distributions of power, "
    "and how those distributions, in turn, affect the system's ability to achieve its stated goals.\n\n"

    "The analytic framework for this work is as follows:\n"
    "- 'Structural features' are manifest at multiple levels.\n" 
    "- The primary level is conceptual framings and assumptions about development and humanitarian action (e.g., racialized, gendered, colonial, "
    "geopolitical, economic, technocratic, etc.).\n"
    "- These framings shape subsequent levels via the design and operation of institutional forms, including (non-exhaustively) governance "
    "structures, financing arrangements, accountability systems, knowledge systems, partnership models, and organizational practices.\n"
    "- These institutional forms give rise to further structural levels that shape the distribution of power within the system, expressed as "
    "(non-exhaustively) decision-making authority, control over resources, and agenda-setting capacity.\n"
    "- Similar distributions of power may persist across different historical periods even when institutional forms change, "
    "where underlying structural framings remain stable.\n\n"

    "We expect the articulation of different levels of structures to appear independently in the text. Not every structural level will be "
    "described in relation to those above and below it. Different authors and different passages will independently talk about 'structures' and invoke different levels.\n\n"

    "Insight extraction should adopt this conceptual framing when determining whether any claim in the corpus is relevant to the questions posed below.\n\n"

    "The examples used above are illustrative and non-exhaustive, they should not prefigure or bias insight extraction.\n\n"

    "HISTORICAL PERIOD REQUIREMENT:\n"
    "Every insight you extract must be attached to at least one historical period. "
    "If no defensible period can be attached to an insight, do not extract it.\n\n"

    "Periods are analytical categories, not simple date ranges. They may overlap in time and are not mutually exclusive. "
    "The text may not use the exact period labels, so assign periods based on the specific insight and the evidence "
    "in the text you are given, not on the overall topic of the paper.\n\n"

    "The predefined periods are:\n"
    "- post WWII reconstruction, development, and decolonization\n"
    "- cold war developmentism\n"
    "- structural adjustment and the debt crisis\n"
    "- post 1990 liberal internationalism and governance\n"
    "- post 9/11 securitization\n"
    "- post 2015 sustainable development\n\n"

    "Rules for assigning periods:\n"
    "- An insight may belong to several periods. If several apply, include all of them.\n"
    "- If an insight refers to a distinct historical framework, initiative, era, administration, program, or aid regime "
    "that the predefined periods do not adequately represent (for example the MDGs, PEPFAR, the Alliance for Progress, "
    "or the Green Revolution), use its name in place of, or alongside, a predefined period.\n"
    "- Use a period when it is explicitly stated in the text, or when it can reasonably be inferred from the content of "
    "the specific insight and the surrounding text.\n"
    "- The publication date is the weakest evidence. Use it only when the insight is clearly about the author's "
    "contemporary present (framed as current, present-day, recent, ongoing, or equivalent) and nothing stronger exists. "
    "Never assign a period merely because the paper has a publication date. If an insight is general, theoretical, or "
    "historical without clear temporal anchoring, do not infer a period from the date, and do not extract it.\n\n"

    "Required format:\n"
    "Begin every insight with its period(s), written in plain words with spaces and never underscores, followed by a colon "
    "and then the insight text. The period prefix is the only permitted addition to the source wording.\n"
    "- One period: 'During <period>: <insight>'\n"
    "- Two periods: 'During <period A> and <period B>: <insight>'\n"
    "- Three or more: 'During <period A>, <period B>, and <period C>: <insight>'\n"
    "Use this exact pattern for every insight, with no variations in wording, punctuation, or capitalisation of 'During'.\n\n"
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

# Initialize an insights dataframe with the question information - this will be updated with the paper metadata and insights as we go through the pipeline
insights_df = questions_df.copy()

# ==========================================================
# 1. INGEST DOCUMENTS
# ==========================================================

# The Ingestor loads documents from disk, extracts text, confirms metadata,
# and breaks the text into chunks suitable for LLM reading.

# Note if you arrived here via the getlit tool, you can initialize the Ingestor class with the corpus_state object created by getlit - downloads.corpus.state
# ingestor = core.Ingestor(
#     corpus_state=downloads.corpus_state,
#     llm_client=llm_client,
#     ai_model="gpt-4o"
# )
# If you are starting with your own corpus, you can initialize the Ingestor with the questions and insights dataframes created above

ingestor = core.Ingestor(
    questions=questions_df,
    papers=insights_df,
    llm_client=llm_client,
    ai_model="gpt-6-sol"
)

# Read PDF/HTML files and extract text
ingestor.ingest_papers()

# Validate metadata against the text (title, author, year)
# Metadata is treated as a first-class object in the pipeline
ingestor.update_metadata()

#---------------
# Pull the metadata from the zotero lib
zot_meta = pd.read_csv(r"C:\Users\jmorrissey\Documents\python_projects\future_of_aid_consolidated\q_set1_all_papers\ReadingMachine\data\corpus\foreign_assistance_readingmachine_sources.csv")

ingestor.corpus_state.insights = (
    zot_meta
    .rename(columns = {
        "Publication Year": "paper_date",
        "Call Number": "paper_id",
        "Author": "paper_author",
        "Title": "paper_title"
    })
    .filter(items = ["paper_date", "paper_id", "paper_author", "paper_title"])
    .merge(
        ingestor.corpus_state.insights.drop(columns = ["paper_author", "paper_title", "paper_date"]), 
        on = "paper_id", 
        how = "left"
    )
)
#-------------


# Drop duplicate files to make sure each paper only enters the pipeline once. This is a two-step process:
# 1) Drop exact duplicates based on exact hash of the full text
# 2) Drop potential duplicates based on a jaccard score calculated on shingles of the text - creates a manual review file
# This is a semi-manual process - the system identifies potential duplicates and you review them to confirm which ones to drop.
ingestor.drop_duplicates()

# ----------------------------------------------------------
# Manual Step
# ----------------------------------------------------------
#
# Review the exported CSV files located in:
#
#    data/potential_duplicates/
#
# Remove any duplicate rows manually.
#
# It can also be useful to manually update any missing metadata fields at this stage. Data might be missing due to the import format.
#
# After editing the CSV files run:
# Have to drop some front and back matter content that does not have an author - write to csv and then update the state
ingestor.corpus_state.insights.dropna(subset =["paper_author", "paper_title", "paper_date"]).to_csv(r"C:\Users\jmorrissey\Documents\python_projects\future_of_aid_consolidated\q_set1_all_papers\ReadingMachine\data\fuzzy_checks\ingest\duplicate_check.csv")

ingestor.update_state("duplicate_check.csv")

# Generate the citations that you want to appear in the final report - this is generated from the metadata - and will be what appears in the final report in the form (author date).
ingestor.gen_unique_citations()

# Break each document into bounded segments (~paragraph scale)
# This ensures the LLM reads manageable text windows
ingestor.chunk_papers()

# Optional - sanity check your chunks to make sure they have not exploded due to OCR issues or something similar
# Useful to do before sending chunks for insight extraction as this is the most expensive step in the pipeline
ingestor.chunk_sanity_check()

# ==========================================================
# Pipeline recovery helper
# ==========================================================

# If a run is interrupted, you can call:
# from readingmachine import utils
# utils.restart_pipeline()
#
# This prints the most recent saved state and tells you how to resume. You may have to also run
# from readingmachine import state
# in order to load the most recent state object into memory before resuming the pipeline


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
    ai_model="gpt-6-sol",
    paper_context=paper_context
)

#--------------

# Extract claims from each chunk
insights_generator.get_chunk_insights()

# Extract higher-level arguments spanning entire documents
# Two papers ingested empty, we have to get rid of these to make the meta insights work
# Get the bad papers
bad_papers = set(insights_generator.corpus_state.full_text["paper_id"]) - set(insights_generator.corpus_state.insights["paper_id"])
# Remove them
insights_generator.corpus_state.full_text = insights_generator.corpus_state.full_text[
    ~insights_generator.corpus_state.full_text["paper_id"].isin(bad_papers)
].reset_index(drop=True)

insights_generator.get_meta_insights()

insights_generator.check_insights()
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
    n_neighbors=5,
    min_dist=0.0,
    n_components=20,
    metric="euclidean",
    random_state=config.seed
)

# ----------------------------------------------------------
# Clustering
# ----------------------------------------------------------

# Tune HDBSCAN clustering parameters
cluster.tune_hdbscan_params(
    min_cluster_sizes=[5, 10, 15, 20],
    metrics=["euclidean", "manhattan"],
    min_sample_ratios=[0.5, 0.25, 0.1, 0.05], 
    cluster_selection_method=["eom", "leaf"]
    )

# Inspect the tuning results a choose params that balance cluster coherence with coverage - question-by-question
cluster.hdbscan_tuning_results.to_html("hdbscan_tuning_results.html")

# Apply clustering
cluster.generate_clusters({
    # Q0: 55 clusters, DB 0.443 (tightest), 25.4% outliers
    "question_0": {
        "min_cluster_size": 20,
        "metric": "manhattan",
        "cluster_selection_method": "eom",
        "min_samples": 10
    },

    # Q1: 42 clusters, DB 0.440 (near-tightest), 17.7% outliers (near-lowest)
    "question_1": {
        "min_cluster_size": 20,
        "metric": "manhattan",
        "cluster_selection_method": "eom",
        "min_samples": 10
    },

    # Q2: 79 clusters (lowest count), DB 0.502, 23.5% outliers
    "question_2": {
        "min_cluster_size": 20,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "min_samples": 10
    },

    # Q3: 43 clusters, DB 0.512, 15.2% outliers (near-lowest)
    "question_3": {
        "min_cluster_size": 15,
        "metric": "manhattan",
        "cluster_selection_method": "eom",
        "min_samples": 8
    }
})
# ==========================================================
# 4. SYNTHESIZE THEMES
# ==========================================================

# The Summarize class performs the thematic synthesis pipeline.

from readingmachine import state
latest_corpus_state = state.CorpusState.load(filepath = r'c:\Users\jmorrissey\Documents\python_projects\ReadingMachine\data\runs\09_clusters')
summarize = core.Summarize(corpus_state=latest_corpus_state, llm_client=llm_client, ai_model="gpt-4o", paper_output_length=10000)


summarize = core.Summarize(
    corpus_state=cluster.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-6-sol",
    paper_output_length=10000, 
    analytical_framework=paper_context
)

# ----------------------------------------------------------
# Cluster summarization
# ----------------------------------------------------------

# Each cluster is summarized individually.
# Clusters are processed in shortest-path order through embedding space
# to improve narrative coherence.

summarize.summarize_clusters()

# Note if your summarize_clusters step produces errors, you should reduce the cluster size parameters:
# 1. If clusters are showing errors you should re-run generate_clusters() method (Clustering class) and lower the hdbscan_cluster_size_cap parameter (default is 1000). 
# 2. If outliers are showing errors upon summarization you should re-run  generate_clusters() method (Clustering class) and lower the outlier_cluster_size_cap parameter (default is 300). 

# ----------------------------------------------------------
# Theme generation
# ----------------------------------------------------------

##------------ TESTING

old_schema = summarize.summary_state.theme_schema_list[-1]

new_schema, repair_plan = summarize._repair_schema()

old_schema.to_dict(orient="records")
repair_plan.to_dict(orient="records")
new_schema.to_dict(orient="records")

##------------ \END TESTING

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

# Iterate until the model indicates that there are no obvious improvements to the schema to be offered. 

# Reduce redundancy across themes
summarize.address_redundancy()

# ==========================================================
# 5. RENDER FINAL OUTPUT
# ==========================================================

renderer = render.Render(
    summary_state=summarize.summary_state,
    corpus_state=summarize.corpus_state,
    llm_client=llm_client,
    ai_model="gpt-6-sol"
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

# Trace a specific claim back to it source text and metadata
# This will need to reflect your specific question and theme text, produced by your run
renderer.trace_claim(
    question_text="How does remote work affect worker satisfaction and retention?",
    theme_label="Work-Life Balance",
    citation_lastname= "Lingfeng",
    citation_year=2021
    )
