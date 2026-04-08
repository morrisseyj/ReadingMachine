# ReadingMachine: Structured Corpus Reading with LLMs

**ReadingMachine is a structured methodology for large-scale thematic synthesis across natural-language corpora.**

It coordinates large language models in a **constrained reading workflow** that extracts structured insights from documents and recombines them into an inspectable synthesis.

Rather than retrieving fragments or repeatedly summarizing documents, ReadingMachine performs a **full analytical pass over a defined corpus**, producing a structured map of the arguments it contains.

ReadingMachine should be understood primarily as a **research methodology implemented through machine reading**, rather than a conventional AI application.

---

## What it does

**Input**
- A corpus of documents (PDF / HTML)
- A set of research questions

**Output**
- A structured thematic synthesis
- Built from **traceable atomic insights**
- With intermediate artifacts preserved (chunks → insights → clusters → themes)

---

## How it works (high level)

```text documents → insights → clusters → themes → synthesis```

- Documents are broken into bounded segments and read systematically
- The system extracts atomic insights (claims) rather than summaries
- Insights are organized semantically and grouped into themes
- Synthesis is performed over structured inputs rather than raw text

The result is a corpus-level map of arguments, with traceability back to source material.

---

## Quick Start

### Recommended setup

Python 3.11

uv

### Install dependencies

```bash
uv sync
```

### Configure API key

Create a .env file:

```bash
OPENAI_API_KEY=your_key_here
```

### Add your corpus

Place documents in:

`data/corpus`

### Run the pipeline

ReadingMachine is designed as an **interactive workflow**, not a single command.

Start from the example script: `/examples/run_core_pipeline.py`

Then execute it step-by-step in a Python environment, following the comments in the script.

The pipeline includes:

- manual review steps (e.g. duplicate detection)
- parameter selection (e.g. clustering)
- iterative refinement (e.g. theme generation)

For a guided walkthrough, see: `/commentary/tutorial.md`

---

## Environment

ReadingMachine uses a pinned environment (`uv.lock`) for reproducibility.

Tested baseline:
- OS: Windows (x86_64)
- Python: 3.11

macOS and Linux are not yet fully validated and may require minor adjustments. Cross-platform compatibility is an active area of development.

---

## Repository Overview

`/data`              → user corpus + generated artifacts  
`/documentation`     → architecture, pipeline, and code docs  
`/examples`          → runnable workflows  
`/evaluation`        → example outputs and evaluation materials  
`/readingmachine`    → core modules  
`/commentary`        → tutorials and explanatory guides  

---

## Documentation

ReadingMachine separates documentation by purpose:

- `/documentation/Whitepaper` → methodology, assumptions, limitations
- `/documentation/PIPELINE.md` → end-to-end workflow
- `/documentation/ARCHITECTURE.md` → system and state design
- `/documentation/CODE_ARCHITECTURE.md` → class structure and execution model

---

## Examples and Tutorials

- `examples/run_core_pipeline.py` → full pipeline run
- `examples/run_getlit_pipeline.py` → optional corpus discovery
- `examples/toy_corpus.md` → small test dataset

For a step-by-step walkthrough, see materials in: `/commentary/tutorial.md`

---

## When to use ReadingMachine

ReadingMachine is designed for:

- literature synthesis
- policy and institutional analysis
- qualitative data mapping
- large-scale document review

It is most useful when:

- coverage matters
- omission is costly
- the goal is structural understanding of a corpus, not question answering

---

## What it is not

ReadingMachine is not:

- a retrieval (RAG) system
- a question-answering system
- an agentic research workflow

It does not reason over a corpus or evaluate claims.

It produces a structured representation of what the corpus contains, leaving interpretation to the researcher or downstream analysis.

---

## Status

ReadingMachine is an experimental methodological framework.

Some aspects—particularly behavior at scale—are based on early observations rather than full empirical validation. The project is released to enable:

- testing
- critique
- iterative refinement

---

## Collaboration

This project is an open methodological experiment.

Contributions are especially valuable in:

- expert evaluation of outputs
- adversarial testing
- benchmarking against other methods
- parameter sensitivity and clustering diagnostics
- large-scale testing across domains
- code hardening and performance optimization

---

## Citation

If you use ReadingMachine in your work, please cite:

Morrissey, J. (2026). ReadingMachine: A computational framework for qualitative corpus synthesis.
https://github.com/morrisseyj/ReadingMachine

---

## Notes for Contributors

- uv.lock defines a known-good environment
- Ensure the pipeline runs end-to-end after dependency changes
- Cross-platform support (macOS/Linux) is still evolving

---

## Acknowledgement

This project was developed as part of research and innovation work at Oxfam America.