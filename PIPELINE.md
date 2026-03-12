# ReadingMachine Pipeline Architecture
                         ┌───────────────────────────┐
                         │        DOCUMENTS          │
                         │        (PDF / HTML)       │
                         └──────────────┬────────────┘
                                        │
                                        ▼
                             ┌──────────────────┐
                             │     Ingestor     │
                             │  (document load) │
                             └─────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   CorpusState   │
                              │                 │
                              │  full_text      │
                              │  chunks         │
                              │  insights       │
                              └────────┬────────┘
                                       │
                                       ▼
                               ┌─────────────────┐
                               │     Insights    │
                               │                 │
                               │ chunk insights  │
                               │ meta insights   │
                               └────────┬────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │     Clustering   │
                              │                  │
                              │ embeddings       │
                              │ dimensionality   │
                              │ reduction        │
                              │ clustering       │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   CorpusState   │
                              │ (insights now   │
                              │  include        │
                              │  clusters)      │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │    Summarize    │
                              └────────┬────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │      SummaryState      │
                           │                        │
                           │ cluster summaries      │
                           │ theme schemas          │
                           │ insight-theme mapping  │
                           │ populated themes       │
                           │ orphan audits          │
                           │ redundancy pass        │
                           └────────┬───────────────┘
                                    │
                                    │
                                    ▼
                         ┌─────────────────────────┐
                         │        Render           │
                         │                         │
                         │ title + exec summary    │
                         │ question summaries      │
                         │ stylized theme text     │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                          ┌────────────────────────┐
                          │      FINAL OUTPUT      │
                          │  Markdown / DOCX / PDF │
                          └────────────────────────┘

---
# Iterative Synthesis Loop

cluster summaries  
        │  
        ▼  
theme schema generation  
        │  
        ▼  
insight → theme mapping  
        │  
        ▼  
theme population  
        │  
        ▼  
orphan detection  
        │  
        ▼  
orphan reintegration  
        │  
        ▼  
(optional) regenerate schema  
        │  
        └─────────────── repeat ────────────────  

---

# State Separation

ReadingMachine maintains two distinct pipeline states.

CorpusState
    documents
    → chunks
    → insights
    → clusters

SummaryState
    cluster summaries
    → theme schemas
    → theme mappings
    → populated themes
    → orphan audits
    → redundancy pass

This separation preserves:

- traceability (insight → chunk → document)  
- inspectability of intermediate artifacts  
- reproducibility of analytical runs.  

