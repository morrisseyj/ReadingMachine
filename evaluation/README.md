# Evaluation of ReadingMachine Outputs

This folder contains materials related to the open evaluation of ReadingMachine outputs.

The aim of this evaluation is to assess the epistemic fidelity of structured corpus reading at scale: whether ReadingMachine produces a faithful, coherent, and usable mapping of what a corpus contains, prior to any downstream reasoning or interpretation.

This evaluation is intentionally not a review of writing quality, persuasiveness, or correctness of conclusions.

## What This Evaluation Is (and Is Not)

### What it is

The evaluation assesses whether ReadingMachine:

- Covers major thematic strands present in the corpus
- Represents arguments and disagreements recognizably and without distortion
- Avoids hallucination or misattribution of sources
- Produces conceptually coherent themes aligned with the research questions
- Preserves conflict and minority positions where they exist
- Produces an output that is usable as a basis for downstream analysis

In short, it asks whether the system has read the corpus faithfully.

### What it is not

This evaluation does not assess:

- Prose style, tone, or readability
- Whether reviewers agree with arguments presented
- Whether conclusions are normatively or politically correct
- Whether the output reflects a reviewer’s preferred framing

Reviewers are explicitly not asked to judge the output as an argument or essay.

## Open, Attributed Review Model

Evaluation is conducted as an open, attributed review process.

Reviews are not anonymous

- Reviewers self‑identify their name, affiliation, and domain expertise
- Attribution is used to prevent misuse and to contextualize feedback, not to police opinion

Participation is voluntary, and disagreement among reviewers is expected and informative.

## Partial Completion Is Expected

Reviewers are not expected to complete the entire evaluation form.

Instead:

- Each evaluation dimension can be assessed independently
- Reviewers are encouraged to answer only those questions they feel qualified to assess
- “Not assessed” is a valid and meaningful response for every dimension

Because no reviewer is expected to know the full corpus exhaustively, *completeness is evaluated within the reviewer’s domain knowledge*, not against an impossible standard of total recall.

## How to Submit a Review

Reviews are submitted via a structured Google Form:

👉 LINK_TO_GOOGLE_FORM

The form is designed to be accessible to researchers who do not work in GitHub while remaining technically precise and methodologically explicit.

## How Evaluation Results Are Used

### Aggregation

Evaluation results are aggregated by dimension, not collapsed into a single score.

For each dimension (e.g. hallucination risk, thematic coherence):

- We report how many reviewers assessed that dimension
- We report the distribution of responses
- We explicitly report the number of reviewers who marked “Not assessed”

This approach reflects the fact that reviewers engage with different parts of the output at different levels of depth.

### Interpretation

Disagreement, uncertainty, and partial assessment are treated as signal, not noise.

Aggregated results are intended to:

- Surface recurring strengths and failure modes
- Indicate areas of synthesis stress or conceptual strain
- Inform further development of the methodology

They are not used to rank or score the system competitively.

## Contents of This Folder

This folder contains the following materials:


- output/
The ReadingMachine output(s) being evaluated, including the full structured synthesis.


- EVAL_INSTRUCTIONS.md
A copy of the evaluation framing and instructions presented to reviewers.


- EVALUATION_SUMMARY.md
Periodically updated aggregation of evaluation results, including:

- response counts per evaluation dimension
- distributions of scores
- selected attributed reviewer comments

Additional supporting materials (e.g. review schema, aggregation notes) may be added over time for transparency.

## Relationship to the ReadingMachine Project

This evaluation process is integral to ReadingMachine’s methodological claims.
ReadingMachine is not presented as producing definitive syntheses or authoritative conclusions. It is presented as a structured reading system whose outputs must themselves be inspectable, contestable, and evaluable.
Open, bounded, attributed evaluation is therefore treated as part of the method, not as an afterthought.

## Questions or Contributions
If you have questions about the evaluation process, or if you wish to engage more deeply with the methodology, please see the main project README or open an issue in the repository.