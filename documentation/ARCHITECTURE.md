# ReadingMachine ARCHITECTURE

# Code Architecture

ReadingMachine is organized into a small set of modules that implement the
structured reading methodology described in the README.

The system separates **corpus discovery**, **corpus processing**, and
**thematic synthesis** into distinct layers.

---

## Repository Structure

readingmachine/  
│  
├── core.py  
│ Implements the primary analytical pipeline:  
│ ingestion → insight extraction → clustering → synthesis  
│  
├── state.py  
│ Defines the two persistent state objects used throughout the system:  
│ CorpusState and SummaryState.  
│  
├── render.py  
│ Generates presentation outputs (Markdown, DOCX, PDF) from the  
│ synthesized thematic structure.  
│  
├── prompts.py  
│ Central registry for all prompts used across the pipeline.  
│  
├── utils.py  
│ Shared utility functions including LLM call wrappers and state validation.  
│  
├── config.py  
│ Centralized configuration for filesystem paths and naming conventions.  
│  
└── getlit/  

Optional corpus discovery tools:  
- search string generation  
- academic literature retrieval  
- grey literature discovery  
- deduplication and corpus assembly  

---

## Pipeline Layers

The system operates in three conceptual layers:

### 1. Corpus Discovery (optional)

Implemented in `getlit/`.

research questions
→ search strings
→ literature retrieval
→ deduplication
→ download preparation

This stage builds the document corpus but is not required if the user
already has a set of documents.

---

### 2. Corpus Processing

Implemented primarily in `core.py`.

documents
→ full text ingestion
→ deduplication
→ chunking
→ insight extraction
→ embedding
→ clustering

This stage converts raw documents into the structured **insight corpus**
stored in `CorpusState`.

---

### 3. Thematic Synthesis

Also implemented in `core.py`.

cluster summaries
→ theme schema generation
→ insight-to-theme mapping
→ theme population
→ orphan detection
→ iterative refinement
→ redundancy reduction

Artifacts from this stage are stored in `SummaryState`.

---

### 4. Rendering

Implemented in `render.py`.

thematic synthesis
→ narrative summaries
→ stylistic rewrite (optional)
→ question summaries
→ executive summary
→ final report generation

Render artifacts are presentation outputs and are intentionally not
treated as analytical state.

---

# State Architecture

## Overview

ReadingMachine maintains two persistent state objects that track the transformation of a corpus through the pipeline:

1. CorpusState   → represents the corpus and extracted insights

2. SummaryState  → represents the interpretive synthesis of those insights

The two objects correspond to two different phases of the method:

1. Corpus processing → extraction and organization of insights

2. Thematic synthesis → clustering, thematic structure, and narrative summaries

Separating these layers preserves both **traceability** and **reproducibility**.

- **CorpusState** records how claims are extracted from the text.
- **SummaryState** records how those claims are organized and synthesized into themes.

Additionally, the rendering layer maintains a small set of persisted artifacts.
These artifacts behave like a pseudo-state but are intentionally not managed
by a dedicated state object.

They reside within the `Render` class and represent **presentation outputs**
rather than analytical state.

---

# CorpusState

CorpusState tracks the transformation of raw documents into structured analytical units.

It contains four primary DataFrames, all persisted to Parquet:

- questions
- full_text
- chunks
- insights

These tables represent successive levels of structure in the corpus.

---

## questions

Contains the research questions guiding the analysis.

Typical columns:

| column | description |
|------|-------------|
| question_id | stable identifier for the research question |
| question_text | full text of the research question |

This table defines the **analytical frame** used during insight extraction.

---

## full_text

Stores the full text of each document.

Typical columns:

| column | description |
|------|-------------|
| paper_id | unique document identifier |
| full_text | extracted full document text |

Additional metadata fields may be included.

---

## chunks

Stores segmented text units derived from the full text.

Typical columns:

| column | description |
|------|-------------|
| paper_id | document identifier |
| chunk_id | unique identifier for the chunk |
| chunk_text | text of the chunk |

Chunks allow bounded reading operations.  
Insight extraction operates primarily on these units.

---

## insights

The insights table is the **core analytical dataset** of the pipeline.

Each row represents an atomic claim extracted from the corpus.

Typical columns include:

| column | description |
|------|-------------|
| insight_id | unique identifier for the insight |
| question_id | research question the insight addresses |
| paper_id | document source |
| chunk_id | originating chunk |
| insight | extracted claim text |
| cluster | cluster assignment |
| search_term | optional (when using `getlit`) |
| paper_author | text str of document authors - should not be a list to prevent parquet persistence issues of embedded lists|
| paper_date | int of the document publishing date |
| paper_title | the document title |
| question_text | the research questions |
| no_author_insight_string | insight string cleaned of citation information to ensure clustering is on insights semantics, not on metadata |
| cluster_prob | estimated likelihood that a insight belongs to the assigned cluster |
| full_insight_embedding | embedding array in its full dimensions |
| reduced_insight_embedding | embedding array in its reduced dimensions |

NOTE: Document **metadata is considered a first class object in the pipeline** for ensuring traceability. Thus the default is to have the user call and LLM to identify the metadata for each paper.

The user can skip this if thier documents or insights are not publications with normal metadata (memos, qualitative survey results etc.) in that case the user is expected to secure thier own metatdat in a format that makes sense. 

---

## Conceptual role

The insights table represents the **structured semantic index of the corpus**.

All downstream analysis operates on this table.

In particular:

- clustering
- theme mapping
- synthesis

are all driven by insights rather than raw text.

---

## Persistence

CorpusState is persisted as a directory of Parquet files.

Each DataFrame is written separately:

- questions.parquet
- full_text.parquet
- chunks.parquet
- insights.parquet


Embeddings are stored using Arrow list types to preserve numeric structure.

---

## Fingerprinting

CorpusState includes a deterministic fingerprint function.

The fingerprint is generated from:

- insight_id
- insight text
- cluster assignments

This allows the pipeline to detect changes in corpus structure and prevent invalid pipeline resumes.

---

# SummaryState

SummaryState records the interpretive artifacts produced during thematic synthesis.

Unlike CorpusState, which stores tidy analytical data, SummaryState stores **pipeline passes**.

Each summarization stage produces a new DataFrame appended to a list.

---

## cluster_summary_list

Contains summaries of each cluster.

Each cluster summary includes:

| column | description |
|------|-------------|
| question_id | unique question identifier |
| question_text | the research question |
| cluster | the cluster number organized along the shortest path between cluster centroids|
| summary | the cluster summary |

Typical length:

1

Cluster summaries are generated once at the start of the synthesis process.
They are not regenerated during theme iteration.

Cluster summaries are used as scaffolding for theme generation. They do not constitute themes themselves.

---

## theme_schema_list

Stores the theme schema generated during each thematic iteration.

Each schema includes:

| column | description |
|------|-------------|
| theme_id | unique theme identifier |
| theme_description | textual description |
| theme_label | textual name for the theme |
| instructions | rules for assigning insights - INCLUDE and EXCLUDE criteria for all theme except 'Conflict' theme in which DETECTION TRIGGERS are defined |
| question_id | unique question identifier |
| question_text | the research question |

**Note** theme_id is strictly type int (flagged on load and save if not int) and globally unique across the dataset. This is important for ordering the final synthesis and allowing for overall inspectability.

---

## mapped_theme_list

Maps insights to themes.

Typical columns:

| column | description |
|------|-------------|
| insight_id | insight identifier |
| theme_id | assigned theme |
| question_id | unique question identifier |

Multiple theme assignments are permitted.

---

## populated_theme_list

Contains the textual summaries of each theme after synthesis.

Typical columns:

| column | description |
|------|-------------|
| thematic_summary | textual summary of the theme |
| question_id | unique question identifier |
| theme_id | unique theme identifier |
| theme_label | textual name for the theme |
| theme_description | textual description |
| question_text | the research question |
| allocated_length | the number of words the user allocated for theme length - estimated from the proportion of insights in the theme, and derived from a user defined total length for the paper |
| current_length | the current number of words in the theme description |
| perc_of_max_length | the percentage of the max allowed length constituted by the current length |
| length_flag | bool (1/0) indicating if current length within 90% of max length |

Each pass corresponds to a thematic synthesis iteration.

Length calculations provide tunable approach by which the user can ensure that the model is not compressing too aggressively during summarization to remain within word count.

---

## orphan_list

Records insights that were not incorporated into the most recent synthesis.

Typical columns:

| column | description |
|------|-------------|
| insight_id | unqiue insight id |
| question_id | unique question identifier |
| theme_id | unique theme identifier |
| found | bool indicating if the insight was identified - all 'false' for orphans |

These are both identifed and forced into the "thematic_summary" of the latest populated_theme_list, in the same pass. This is done to limit omission risk and improve theme schema through iteration. 

---

## redundancy_list

Stores the final redundancy-corrected synthesis.

Typical columns are the same as populated_theme_list, however the thematic_summary variable now reflects the content without redundancy

Typical length:

1

Redundancy pass is only undertaken once, **after all iterating on orphans and theme schema development has completed**. It is an optional pass. The user can choose to prioritize a redundant but higher fidelity output and just pass the output of the final orphan pass to the render class.

---

# Pipeline Iteration Model

Theme synthesis proceeds iteratively:

cluster summaries
→ theme schema
→ insight mapping
→ theme population
→ orphan detection
→ iteration (pass populated themes with orphans inserted back to theme schema generation)


Each iteration produces a new entry in the corresponding lists.

---

# Render Artifacts (Pseudo-State)

The rendering stage produces a small number of additional artifacts that are
persisted for reuse but are not part of the core analytical pipeline.

Unlike `CorpusState` and `SummaryState`, these artifacts are stored directly
within the `Render` class rather than being managed by a dedicated state object.

This design choice avoids introducing additional architectural complexity for
objects that do not evolve across pipeline iterations.

## Persisted render artifacts

The rendering layer currently produces three persistent artifacts.

### Title and Executive Summary

Generated from the complete synthesized corpus representation.

Stored as:

title_exec_summary

Typical contents:

| column | description |
|------|-------------|
| content | generated text |
| doc_attr | `"title"` or `"exec_summary"` |

---

### Question Summaries

High-level narrative summaries of the thematic results for each research question.

Stored as:

question_summary_df


Typical contents:

| column | description |
|------|-------------|
| question_id | research question identifier |
| question_text | research question text |
| content | generated summary |
| doc_attr | `"question_summary"` |

---

### Stylized Theme Rewrite

Optional stylistic rewriting of theme summaries to improve readability.

This does not overwrite the original synthesis output.  
Instead, the stylized text is appended as an additional column.

Stored as:

stylized_rewrite_df


Typical contents:

| column | description |
|------|-------------|
| question_id | research question identifier |
| theme_id | theme identifier |
| stylized_text | rewritten theme text |

The original `thematic_summary` remains preserved.

---

## Pipeline Stage Ordering

The summarization pipeline follows a strict stage order:

cluster summaries  
→ theme schema  
→ insight mapping  
→ theme population  
→ orphan detection  
→ iteration  
→ redundancy handling  
→ rendering

Each stage depends on artifacts generated by the preceding stage.

This ordering is enforced by the `SummaryState` and `Summarize` classes.  
If a user attempts to execute stages out of sequence, the pipeline will
either:

- prompt the user to overwrite or resume from the appropriate stage, or
- automatically rewind later artifacts to maintain a valid pipeline state.

A `force` flag exists to bypass these safeguards during testing and
development. This flag is not intended for normal analytical workflows.

---

## Render artifact persistence

These artifacts are persisted to Parquet files in the render directory.

Persistence serves two purposes:

1. Avoid re-running expensive LLM calls.
2. Allow rendering to be resumed or modified without regenerating summaries.

Unlike the core pipeline states, render artifacts are **terminal outputs** and
do not participate in iterative refinement.

Artifacts are compiled into a single dataframe for rendering via a dedicated function within the Render class.

---

## Hash validation

The render stage validates that the summaries being rendered match the
summaries used when the render artifacts were created.

This is done using a deterministic hash of the `summary_to_render` DataFrame.

If the hash differs from the stored value, the render pipeline will prompt the
user before overwriting previous artifacts.

---

## Conceptual role

Render artifacts represent **presentation-level outputs** rather than analytical
state.

They exist to support:

- report generation
- stylistic rewriting
- executive summaries

They do not modify the underlying corpus or thematic synthesis.

---

# State Invariants

The following properties must hold for the pipeline to function correctly.

---

## Insight Identity

Every insight must retain a stable identity:

- insight_id
- question_id
- paper_id
- chunk_id


This ensures traceability from synthesis back to the corpus.

The `insight_id` is the primary key for the analytical pipeline and uniquely identifies a single extracted claim.

The identifier must satisfy the following invariants.

---

### Uniqueness

Each row in the `insights` table must have a unique `insight_id`.

No two insights may share the same identifier.

This guarantees that each analytical unit in the corpus can be uniquely referenced throughout the pipeline.

---

### Stability

Once created, an `insight_id` must never change.

Insight identifiers remain stable across all downstream transformations, including:

- embedding
- clustering
- theme mapping
- thematic synthesis
- rendering

This stability ensures that insights remain traceable even as additional attributes are appended during later pipeline stages.

---

### Traceability

Each `insight_id` must correspond to exactly one location in the corpus.

The traceability chain is:

insight_id
→ chunk_id
→ paper_id
→ document text

This structure ensures that any synthesized claim can be traced back to the original text segment from which it was extracted.

Traceability is a core design requirement of ReadingMachine and supports citation-anchored synthesis and claim verification.

---

### Non-Reuse

Insight identifiers are never recycled.

If insights are regenerated (for example after modifying prompts, chunking rules, or corpus composition), a new `CorpusState` should be created rather than reassigning existing `insight_id` values.

This prevents ambiguity in downstream joins and avoids corruption of analytical lineage.

---

### Role in the Pipeline

The `insight_id` functions as the central join key linking the corpus representation to all subsequent analytical structures.

insights table
→ cluster assignments
→ theme mappings
→ orphan detection
→ render traceability

Because all downstream artifacts reference `insight_id`, the integrity of this identifier is critical to maintaining reproducibility and analytical transparency.

---

### Creation Point

`insight_id` values are generated during the **insight extraction stage**, when atomic claims are first produced from the corpus.

They are not generated during clustering, theme mapping, or synthesis.

This reflects the architectural principle that **the extracted insight is the fundamental analytical unit of the pipeline**, and all subsequent transformations operate on these units rather than redefining them.

---

## Canonical Questions

Each `question_id` must correspond to exactly one canonical `question_text`.

The method `enforce_canonical_question_text()` enforces this invariant.

---

## Traceability

Every synthesized claim must be traceable through:

theme
→ insight
→ chunk
→ document

This enables citation-anchored synthesis.

---

## Insight Persistence

Insights are append-only.

Insights may gain additional attributes (clusters, embeddings), but existing insights are never deleted.

---

## Corpus Immutability

Once the corpus has been ingested and insights generated, the corpus
representation should be treated as immutable for the duration of the
analysis.

Documents, chunks, and insights should not be modified after extraction.
Any change to the corpus (for example adding papers, modifying chunking
rules, or regenerating insights) should result in the creation of a new
`CorpusState`.

This invariant ensures that downstream analytical results remain
reproducible.

---

## Cluster Role

Clusters serve as **organizational scaffolding**.

They are not treated as analytical conclusions.

---

## Theme Identity

Within a synthesis pass, `theme_id` must uniquely identify a theme.

---

## Deterministic Persistence

Both state objects must be serializable and reloadable without altering analytical results.

---

# Future Extensions

The state architecture is designed to support future extensions including:

- hierarchical themes
- insight deduplication
- multi-corpus comparisons
- cross-model discourse analysis

These extensions will build on the existing state representation.

---

# Notes for Collaborators

This document describes the conceptual structure of the pipeline state.

Implementation details may evolve, but the invariants described here should remain stable.

---
