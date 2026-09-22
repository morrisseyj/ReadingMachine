# Research Runs

This directory contains selected outputs from ReadingMachine research runs.

Each subdirectory represents one application of the method to a defined corpus and set of research questions. These runs are included to make the development and use of ReadingMachine more transparent, and to provide concrete examples of the kinds of corpus readings the system produces.

The files included here are intentionally selective. They generally include the generated output and source list for each run, but not the full set of intermediate artifacts produced during iterative processing. ReadingMachine generates multiple intermediate layers during theme construction, mapping, population, orphan detection, and schema revision; preserving all of these artifacts would make the repository difficult to navigate. Where relevant, run-level notes are included in the output foreword or the run-specific README.

## Contents

Each run directory may include:

- `README.md`: short description, status, and software version information
- `output.md`: generated corpus reading or synthesis output
- `sources.csv`: source list used for the run

## Versioning

Runs may have been generated using either a tagged release or a development branch. Where possible, each run README identifies the relevant ReadingMachine version, branch, or commit.

The GitHub release tags preserve exact repository states for major methodological versions. The living whitepaper in `/documentation/white_paper.md` describes the current version of the method, while the frozen arXiv paper is preserved separately under `/commentary`.

## Status

These outputs should be read as corpus readings: structured representations of what is present in a defined corpus under specified research questions. They are not, by themselves, position papers, policy recommendations, or claims about what is true outside the corpus.