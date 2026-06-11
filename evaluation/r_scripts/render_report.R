library(rmarkdown)

library(rmarkdown)

# Render HTML directly to eval_output
rmarkdown::render(
  input = "evaluation/r_scripts/public_eval_render.Rmd",
  output_format = "html_document",
  output_file = "../eval_output/public_eval.html"
)

# Render GitHub Markdown in-place, next to the Rmd
rmarkdown::render(
  input = "evaluation/r_scripts/public_eval_render.Rmd",
  output_format = "github_document",
  output_file = "public_eval.md"
)

# Move Markdown output and generated figure folder to eval_output
if (dir.exists("evaluation/eval_output/public_eval_files")) {
  unlink("evaluation/eval_output/public_eval_files", recursive = TRUE)
}

if (file.exists("evaluation/eval_output/public_eval.md")) {
  unlink("evaluation/eval_output/public_eval.md")
}

file.rename(
  from = "evaluation/r_scripts/public_eval.md",
  to = "evaluation/eval_output/public_eval.md"
)

if (dir.exists("evaluation/r_scripts/public_eval_files")) {
  file.rename(
    from = "evaluation/r_scripts/public_eval_files",
    to = "evaluation/eval_output/public_eval_files"
  )
}