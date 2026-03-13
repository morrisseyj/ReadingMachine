# Toy Corpus for ReadingMachine

This toy corpus provides a small set of open-access academic papers that can be used to test the ReadingMachine pipeline. This corpus is provided only for demonstration purposes so that users can run the example pipeline without assembling their own document collection.

The corpus focuses on **remote work and productivity**, a topic chosen because:

- the literature is widely accessible
- papers contain clear empirical claims
- documents chunk well for LLM reading
- the topic is easy for new users to understand

These papers (see below) are **not required for the library**. They simply provide a convenient test set so that users can run the example pipeline without assembling their own corpus.

---

# Setup

Download the five papers listed below and place them in:

data/corpus/

Example directory structure:

data/  
└─ corpus/  
├─ paper_1.pdf  
├─ paper_2.pdf  
├─ paper_3.pdf  
├─ paper_4.pdf  
└─ paper_5.pdf  


The filenames **do not strictly need to match the paper IDs**, but keeping them consistent makes it easier to trace documents during testing.

## Environment Setup

ReadingMachine includes a `pyproject.toml` file that defines the required
dependencies.

The recommended way to set up the environment is to use **UV**, which will
create a reproducible virtual environment and install the pinned
dependencies.

### Using UV (recommended)

Install UV if it is not already installed:

https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Then initialize the environment:

```uv sync```

This will install all dependencies specified in `pyproject.toml`.

### Using pip (alternative)

If you prefer to manage the environment manually, you can install the
dependencies using `pip`:

pip install -r requirements.txt

or install the required packages individually based on the
`pyproject.toml`.

---

## Running the Example Pipeline

Once the environment is set up and the toy corpus has been downloaded, you can explore the example workflow contained in:

examples/run_core_pipeline.py

This script demonstrates the full ReadingMachine workflow:

documents → insights → clusters → themes → synthesis → report

---

# Toy Corpus Papers

## Paper 1

**Title**  
Does Working from Home Work? Evidence from a Chinese Experiment

**Authors**  
Nicholas Bloom, James Liang, John Roberts, Zhichun Jenny Ying

**Year**  
2015

**Journal / Source**  
Quarterly Journal of Economics

**DOI**  
https://doi.org/10.1093/qje/qju032

**PDF**  
https://www.nber.org/system/files/working_papers/w18871/w18871.pdf


---

## Paper 2

**Title**  
Why Working From Home Will Stick

**Authors**  
Jose Maria Barrero, Nicholas Bloom, Steven J. Davis

**Year**  
2021

**Source**  
National Bureau of Economic Research

**DOI**  
https://doi.org/10.3386/w28731

**PDF**  
https://www.nber.org/system/files/working_papers/w28731/w28731.pdf


---

## Paper 3

**Title**  
How Effective Is Telecommuting? Assessing the Status of Our Scientific Findings

**Authors**  
Tammy D. Allen, Timothy D. Golden, Kristen M. Shockley

**Year**  
2015

**Journal**  
Psychological Science in the Public Interest

**DOI**  
https://doi.org/10.1177/1529100615593273

**PDF**  
https://journals.sagepub.com/doi/pdf/10.1177/1529100615593273


---

## Paper 4

**Title**  
How Many Jobs Can Be Done at Home?

**Authors**  
Jonathan I. Dingel, Brent Neiman

**Year**  
2020

**Source**  
Journal of Public Economics

**DOI**  
https://doi.org/10.1016/j.jpubeco.2020.104235

**PDF**  
https://bfi.uchicago.edu/wp-content/uploads/BFI_WP_2020-46.pdf


---

## Paper 5

**Title**  
Work-from-anywhere: The Productivity Effects of Geographic Flexibility

**Authors**  
Prithwiraj Choudhury, Cirrus Foroughi, Barbara Larson

**Year**  
2021

**Journal**  
Management Science

**DOI**  
https://doi.org/10.1287/mnsc.2020.3848

**PDF**  
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3304994


---

# Suggested Research Questions

The example pipeline uses the following research questions:

1. What effects does remote work have on worker productivity?

2. How does remote work affect worker satisfaction and retention?

3. What organizational practices enable successful remote work?

4. What constraints limit the adoption of remote work?

5. How has the role of remote work changed over time?

These questions are intentionally broad so that the toy corpus produces:

- multiple insight clusters
- cross-paper synthesis
- visible thematic structure

---

# Notes

The toy corpus is intentionally small. It exists only to demonstrate the workflow:

documents
→ chunks
→ insights
→ clusters
→ themes
→ synthesis


Real ReadingMachine analyses are expected to operate on **much larger corpora**.