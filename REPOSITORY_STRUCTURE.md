# Repository Structure

This repository contains three main components:

1. **The ReadingMachine library**
2. **Documentation describing the architecture**
3. **Example workflows demonstrating usage**

Understanding the structure of the repository can help you quickly
navigate the project.

---

# Library Code

readingmachine/

This directory contains the Python library implementing the
ReadingMachine methodology.

Core modules include:

- `core.py` – document ingestion, insight extraction, clustering, and synthesis
- `state.py` – persistent pipeline state objects
- `render.py` – report generation
- `utils.py` – helper functions
- `config.py` – configuration paths and parameters
- `prompts.py` – LLM prompt templates

Optional corpus discovery tools are located in:

readingmachine/tools

---

# Documentation

documentation/


These files describe the architecture and design of ReadingMachine.

- `ARCHITECTURE.md` – pipeline state model
- `PIPELINE.md` – workflow diagrams
- `CODE_ARCHITECTURE.md` – class interaction overview

---

# Examples

examples/


Example workflows demonstrating how to use the library.

- `run_core_pipeline.py` – full ReadingMachine synthesis workflow
- `run_getlit_pipeline.py` – optional literature discovery workflow
- `toy_corpus.md` – small test corpus for experimentation

---

# Data Directory

data/


Working directory for pipeline data.

This directory is **ignored by git** and used only at runtime.

Structure:

data/
corpus/ # place documents here
pickles/ # intermediate LLM artifacts
runs/ # pipeline state checkpoints

---

# Output Directory

outputs/


Final rendered reports are written here.

Supported formats:

- Markdown
- DOCX
- PDF

---

# Typical Workflow

Most users will:

1. Place documents in:

data/corpus

2. Run the example pipeline:

examples/run_core_pipeline.py

3. Inspect the outputs in:

outputs/

---

# Optional Workflow: Literature Discovery

If you want help identifying literature before building a corpus see:

examples/run_getlit_pipeline.py

This workflow:

research questions
→ search strings
→ academic literature retrieval
→ grey literature retrieval
→ duplicate detection
→ corpus preparation


---

# Notes

ReadingMachine is designed as a **methodological toolkit** rather than
a command-line application. The example scripts are intended to be
read and executed interactively to understand the workflow.