
# Load packages
library(googlesheets4)
library(dplyr)
library(ggplot2)
library(gt)
library(rlang)


familiarity_levels <- c(
  "Deep expertise",
  "Substantial working knowledge",
  "Moderate familiarity",
  "Limited familiarity"
)

plot_eval <- function(data,
                      eval_var,
                      comment_var,
                      possible_evals,
                      top_eval,
                      familiarity_var = "Familiarity with the literature") {
  
  eval_sym <- rlang::sym(eval_var)
  comment_sym <- rlang::sym(comment_var)
  familiarity_sym <- rlang::sym(familiarity_var)
  
  plot_data <- data %>%
    select(author_id, !!familiarity_sym, !!eval_sym) %>%
    mutate(
      familiarity_clean = dplyr::case_when(
        is.na(!!familiarity_sym) |
          trimws(as.character(!!familiarity_sym)) == "" ~ "No response",
        as.character(!!familiarity_sym) %in% familiarity_levels ~
          as.character(!!familiarity_sym),
        TRUE ~ "Other"
      ),
      eval_clean = dplyr::case_when(
        is.na(!!eval_sym) |
          trimws(as.character(!!eval_sym)) == "" ~ "No response",
        as.character(!!eval_sym) %in% possible_evals ~
          as.character(!!eval_sym),
        TRUE ~ "Other"
      ),
      familiarity_clean = factor(
        familiarity_clean,
        levels = c(familiarity_levels, "Other", "No response")
      ),
      eval_clean = factor(
        eval_clean,
        levels = c(possible_evals, "Other", "No response")
      )
    ) %>%
    count(familiarity_clean, eval_clean, name = "n")
  
  plot <- ggplot(plot_data, aes(x = eval_clean, y = n)) +
    geom_col() +
    facet_wrap(~ familiarity_clean) +
    labs(
      x = NULL,
      y = "Number of reviews",
      title = eval_var
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  issue_table <- data %>%
    transmute(
      Reviewer = author_id,
      Familiarity = dplyr::case_when(
        is.na(!!familiarity_sym) |
          trimws(as.character(!!familiarity_sym)) == "" ~ "No response",
        TRUE ~ as.character(!!familiarity_sym)
      ),
      Rating = dplyr::case_when(
        is.na(!!eval_sym) |
          trimws(as.character(!!eval_sym)) == "" ~ "No response",
        TRUE ~ as.character(!!eval_sym)
      ),
      Comment = as.character(!!comment_sym)
    ) %>%
    filter(
      Rating != top_eval,
      Rating != "No response",
      !is.na(Comment),
      trimws(Comment) != ""
    ) %>%
    arrange(Familiarity, Rating, Reviewer) %>%
    gt::gt()
  
  list(
    plot = plot,
    issues = issue_table
  )
}
