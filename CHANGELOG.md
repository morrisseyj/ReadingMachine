# Changelog

All notable methodological and software changes to ReadingMachine are documented here.

## [v0.3.0] - TBD

### Added
- A separate insight check which validates a one shot generated insight as contributing to answering the question. Done to address potentially excessively broad insight interpretation when the model was assessing relevance from a large number of questions.
- Updated the schema generation process to pass a repair history (rather than the full populated schema history). Done to address the fact that in multiple iterations of the synthesis process, passing the full populated schema history would exceed input limits

### Changed

- Updated the module to call the LLM to mirror the new response API from OpenAI
- Batched citation checking with validation and deterministic insertion. Done to address errors in insertion, and the fact that repetition of long citation lists would break the output context window. 
- Made the use of organizing principle for themes a default that can't be opted out of. Done because its performant and making it standard removed conditional complexity from prompts and execution code.
- Updated the schema optimization pass so that it is decomposed into plan optimizaton and and implement plan, mirroring the split in responsibilities for schema repair.

## [v0.2.0] - 2026-09-22

### Added

- Added support for an organizing rationale/proposition in theme schema generation.
- Added downstream use of theme rationale during schema repair, optimization, and theme population.

### Changed

- Updated the living whitepaper to describe the current post-arXiv methodology.
- Clarified versioning relationship between the frozen arXiv paper and the evolving software/method.

### Notes

- This version corresponds to the post-arXiv theme-rationale branch.
- Some outputs generated before this release may reference the `link-theme-rationale` branch directly; the relevant functionality is included in this release.

## [v0.1.0] - 2026-09-22

### Notes

- Retrospective release tag for the repository state associated with the arXiv paper, posted in June 2026.
- Reference version associated with the arXiv paper and initial industrial policy demonstration.
- The frozen scholarly paper is preserved separately under `/commentary`.
- This tag preserves the repository state before post-arXiv methodological updates were merged.