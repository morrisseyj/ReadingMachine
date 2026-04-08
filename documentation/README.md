# Documentation

This folder contains the core conceptual and technical documentation for ReadingMachine.
Together, these documents describe the system from complementary perspectives: high-level methodology, pipeline design, system architecture, and implementation structure.

The goal of this folder is to make the system inspectable, reproducible, and extensible by clearly separating:

- conceptual method (what the system does and why)
- pipeline logic (how data transforms step-by-step)
- architectural design (how components are organized)
- code structure (how the implementation is laid out)

---

## Contents

### whitepaper.md

**Purpose:** Conceptual and methodological foundation

This document introduces ReadingMachine as a computational methodology for structured corpus reading. It explains:

- the problem of large-scale synthesis
- limitations of existing approaches (RAG, summarization, agentic workflows)
- the core design principle: bounded reading operations
- the full methodological pipeline (insights → clusters → themes → synthesis)
- trade-offs (coverage vs cost, abstraction vs traceability)
- empirical behavior from a large-scale run
- implications for research, policy, and knowledge work

**Role in the folder:**

Provides the “why” of the system.  
All other documents implement or operationalize the ideas introduced here.

---

### PIPELINE.md

**Purpose:** End-to-end data flow

This document describes the sequential transformation of data through the system.

It defines the full pipeline:

documents → ingestion → chunking → insights → embeddings → clustering → thematic synthesis → rendering → final outputs

Key elements:

- explicit stage-by-stage flow
- iterative synthesis loop:
    - theme generation
    - mapping
    - orphan detection
    - re-theming
- separation of state:
    - CorpusState (documents → insights → clusters)
    - SummaryState (themes → synthesis outputs)

**Role in the folder:**

Provides the “how data moves” view of the system.

---

### ARCHITECTURE.md

**Purpose:** System design and state model

This document defines the internal structure of the system, including:

- repository layout (core.py, state.py, render.py, etc.)
- separation of layers:
    - corpus discovery (optional)
    - corpus processing
    - thematic synthesis
    - rendering
- detailed specification of:
    - CorpusState
    - SummaryState
- schema definitions for all core tables:
    - questions, chunks, insights, etc.
- persistence model (Parquet-based)
- pipeline invariants:
    - insight identity and traceability
    - corpus immutability
    - deterministic state behavior

It also introduces the concept of render artifacts as pseudo-state, distinct from analytical state.

**Role in the folder:**

Provides the “what exists” view of the system—data structures, invariants, and persistence.

---

### CODE_ARCHITECTURE.md

**Purpose:** Class structure and execution model

This document describes how the system is implemented in code.

It includes:

- class interaction diagram
    - mapping of responsibilities:
    - Ingestor → document processing
    - Insights → insight extraction
    - Clustering → embeddings and grouping
    - Summarize → thematic synthesis
    - Render → output generation
- mutation-based execution model:
    - classes mutate shared state objects
    - no internal orchestration layer
- explicit execution sequence example

It also highlights a key architectural property:

- The system maintains two independent histories:

    - corpus history (documents → insights → clusters)
    - synthesis history (clusters → themes → summaries)

**Role in the folder:**

Provides the “how code is organized and executed” view of the system.

---

How These Documents Fit Together

The four documents are designed to be read together, each adding a layer of understanding:

Layer	Document	Focus
Conceptual	whitepaper.md	Methodology and rationale
Pipeline	PIPELINE.md	Data transformations
System Design	ARCHITECTURE.md	State, schemas, invariants
Implementation	CODE_ARCHITECTURE.md	Classes and execution

A useful reading order is:

- Whitepaper → understand the method
- Pipeline → understand the flow
- Architecture → understand the data model
- Code Architecture → understand implementation

---

Design Principles Reflected in This Documentation

Across all files, the system is built around a consistent set of principles:

- Separation of reading and synthesis
- Delayed compression
- Full-corpus coverage
- Traceability (theme → insight → chunk → document)
- Explicit intermediate representations
- Reproducibility through structured state
- Iterative refinement with omission detection (orphans)

---

### Scope

This folder documents:

- the analytical method
- the pipeline structure
- the system architecture
- the implementation model

It does not cover:

- installation or setup
- usage examples or tutorials

Those can be found in `/README`, `/examples`, and `/commentary`

---

### Notes

The system is experimental and evolving; implementation details may change.
The state model and invariants described in ARCHITECTURE.md should be treated as stable design constraints.
The documentation prioritizes inspectability over brevity, mirroring the philosophy of the system itself.