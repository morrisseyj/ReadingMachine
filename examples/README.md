# Examples

This directory contains example workflows demonstrating how to use
ReadingMachine.

ReadingMachine supports two primary workflows:

1. **Corpus discovery** – building a literature corpus from search engines (useful for literature review type workflows)
2. **Corpus analysis** – performing structured thematic synthesis on a set of documents (supports literature reviews, but can process any type of corpus)

Not all users need both steps.

---

# Example Workflows

## 1. Corpus Discovery (optional)

File:

examples/run_getlit_pipeline.py

Purpose:

Build a literature corpus using the **getlit** tools.

Steps demonstrated:

research questions
→ search strings
→ academic literature retrieval
→ grey literature retrieval
→ duplicate detection
→ AI literature completeness check
→ download architecture

Output:

CorpusState containing candidate papers

After downloading the papers you can proceed to the core pipeline.

NOTE:
Search results from getlit are intentionally broad. Users are expected to
actively make judgement calls on whether papers are relevant before downloading and ingestion. 
This preserves transparency in the retreival process of corpus construction and avoids hidden filtering biases.

---

## 2. Corpus Reading and Synthesis (core workflow)

File:

examples/run_core_pipeline.py

Purpose:

Run the full ReadingMachine analysis on a document corpus.

Pipeline:

documents
→ chunking
→ insight extraction
→ clustering
→ theme generation
→ thematic synthesis
→ report rendering

Input:

PDF / HTML documents placed in:

data/corpus/

---

# Toy Corpus for Testing

File:

examples/toy_corpus.md

This file provides a small open-access corpus that can be used to test the
pipeline without assembling your own dataset.

Download the listed papers and place them in:

data/corpus/

---

# Typical Usage

### If you already have a corpus

Consult only the core pipeline:

examples/run_core_pipeline.py

Running the example script over the toy corpus will cost about $6.50 in OpenAI API credits to run.

### If you want help assembling a literature corpus

Consult the discovery workflow first:

examples/run_getlit_pipeline.py

Running the example gitlit script will cost about $1.75 in OpenAI API credits to run.

Then proceed to:

examples/run_core_pipeline.py

---

# Notes

The example scripts are **annotated workflows**, not CLI commands.

They are intended to be read and executed interactively to understand how
the pipeline operates.