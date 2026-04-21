# Industrial Policy Previous Runs

This folder contains previous run's of ReadingMachine over the Industrial Policy corpus. 

Runs are named with version (v) numbers in the file name. 

## v1 notes

v1 was run on a insights generation prompt that was stricter than the current implementation, and a meta-insights prompt that was looser than the current implementation. The result was a much higher ratio of meta-insights to insights (10450:5633). THe result was a smoother narrative (as compression had happened early for a large portion of the insights), but with less granularity and greater omission risk. v1 also had showed fewer scaling challenges as the insight heterogeneity was less. Finally v1 used an older iterative implementation. Instead of running until the model deemed the schema stable, v1 ran twice (once after cluster formation and once after orphan insertion) before the run was stopped. This was part of an older heuristic for stopping after the schema generation process had seen all of the insights inserted into the themes. Note, in this setup the last step was orphan insertion - so a complete iteration was completed after the second schema gen pass. 