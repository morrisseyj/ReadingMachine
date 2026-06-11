library(rmarkdown)

# Render HTML
rmarkdown::render(
  input = "evaluation/r_scripts/public_eval_render.Rmd",
  output_format = "html_document",
  output_file = "public_eval.html",
  output_dir = "evaluation/eval_output",
  knit_root_dir = "evaluation/r_scripts"
)

# Render GitHub Markdown
rmarkdown::render(
  input = "evaluation/r_scripts/public_eval_render.Rmd",
  output_format = "github_document",
  output_file = "public_eval.md",
  output_dir = "evaluation/eval_output",
  knit_root_dir = "evaluation/r_scripts"
)