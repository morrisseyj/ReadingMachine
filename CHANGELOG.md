# Changelog

All notable methodological and software changes to ReadingMachine are documented here.

## [v0.3.0] - TBD

### Added

### Changed

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