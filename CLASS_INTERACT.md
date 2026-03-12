# Class interaction diagram

                     ┌─────────────────────────────┐
                     │        CorpusState          │
                     │                             │
                     │  questions                  │
                     │  full_text                  │
                     │  chunks                     │
                     │  insights                   │
                     └───────────────▲─────────────┘
                                     │
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              │                      │                      │
      ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
      │    Ingestor    │     │     Insights   │     │    Clustering  │
      │                │     │                │     │                │
      │ reads files    │     │ chunk insights │     │ embeddings     │
      │ html/pdf parse │     │ meta insights  │     │ UMAP reduction │
      │ metadata LLM   │     │                │     │ HDBSCAN        │
      └───────┬────────┘     └───────┬────────┘     └───────┬────────┘
              │                      │                      │
              │ writes               │ writes               │ writes
              ▼                      ▼                      ▼
        full_text              insights + ids        clusters + vectors



                         ┌─────────────────────────────┐
                         │        SummaryState         │
                         │                             │
                         │ cluster_summary_list        │
                         │ theme_schema_list           │
                         │ mapped_theme_list           │
                         │ populated_theme_list        │
                         │ orphan_list                 │
                         │ redundancy_list             │
                         └───────────────▲─────────────┘
                                         │
                                         │
                                         │
                                 ┌───────────────┐
                                 │   Summarize   │
                                 │               │
                                 │ cluster sum   │
                                 │ schema gen    │
                                 │ insight map   │
                                 │ theme pop     │
                                 │ orphan loop   │
                                 │ redundancy    │
                                 └───────┬───────┘
                                         │
                                         │ reads + writes
                                         ▼


                                ┌───────────────────┐
                                │       Render      │
                                │                   │
                                │ title             │
                                │ exec summary      │
                                │ question summary  │
                                │ stylized rewrite  │
                                └─────────┬─────────┘
                                          │
                                          ▼

                                ┌───────────────────┐
                                │    Final Output   │
                                │                   │
                                │ Markdown          │
                                │ DOCX              │
                                │ PDF               │
                                └───────────────────┘

# Mutation Flow (Important Concept)

Classes do not orchestrate the pipeline internally.
Instead they mutate state objects sequentially.

Typical usage pattern:

```
corpus = CorpusState.load(...)

Ingestor(...).ingest_papers()
Insights(...).get_chunk_insights()
Insights(...).get_meta_insights()

Clustering(...).embed_insights()
Clustering(...).reduce_dimensions()
Clustering(...).generate_clusters()

summ = Summarize(...)
summ.summarize_clusters()
summ.gen_theme_schema()
summ.map_insights_to_themes()
summ.populate_themes()
summ.address_orphans()
summ.address_redundancy()

Render(...).render()
```

---

## Why this architecture?

The design intentionally separates:

Corpus processing

Handled by: 

- Ingestor  
- Insights
- Clustering

These stages transform: 

documents → insights → embeddings → clusters

Stored in CorpusState.

---

## Interpretive synthesis

Handled by:

Summarize

These stages transform:

clusters → themes → synthesis

Stored in SummaryState.

---

## Presentation layer

Handled by:

Render

This stage produces reports but does not alter analytical state.

---

## Key Architectural Property

The system preserves two independent histories:

Corpus history
    document → chunk → insight → cluster

Synthesis history
    cluster summary → schema → mapping → themes → orphans → redundancy

This separation is what enables:

- inspectability  
-- reproducibility  
- iterative refinement of themes

without mutating the underlying corpus representation.