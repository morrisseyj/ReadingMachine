# Load packages
library(googlesheets4)
library(dplyr)
library(ggplot2)
library(gt)
library(rlang)

# Set sheet id
sheet_id <- "1Qj9Rp-31E_ZPrXYSxIWbl8wF1NMRkbFlez2LP-suaJM"

# First interactive login
gs4_auth(
  path = "google_sheets_service_account.json",
  cache = TRUE
)

raw_responses <- read_sheet(sheet_id, sheet = 1)

anonymized_responses <- raw_responses %>%
  mutate(
    author_id = case_when(
      `How would you like your comments to be attributed` == "Name, affiliation and role" ~
        paste(Name, Affiliation, Role, sep = ", "),
      `How would you like your comments to be attributed` == "Affiliation and role only" ~
        paste(Affiliation, Role, sep = ", "),
      TRUE ~ "Anonymous"
    )
  ) %>%
  select(-Email)