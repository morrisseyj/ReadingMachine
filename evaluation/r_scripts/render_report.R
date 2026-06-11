library(rmarkdown)

rmarkdown::render(
  input = "public_eval_render.Rmd",
  output_format = "html_document",
  output_file = "../eval_output/public_eval.html"
)

rmarkdown::render(
  input = "public_eval_render.Rmd",
  output_format = "github_document",
  output_file = "../eval_output/public_eval.md"
)