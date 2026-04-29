library(ggplot2)
library(ggforce)
library(dplyr)
library(stringr)

# -----------------------------
# Data
# -----------------------------

steps <- tibble::tribble(
  ~id, ~label, ~stage, ~in_loop,
  1,  "Research\nQuestions",            "Framing", FALSE,
  2,  "Corpus\nIngestion",              "Reading", FALSE,
  3,  "Chunking",                       "Reading", FALSE,
  4,  "Insight\nExtraction",            "Reading", FALSE,
  5,  "Clustering",                     "Organization", FALSE,
  6,  "Cluster\nSummarization",         "Organization", FALSE,
  7,  "Theme\nSchema",                  "Organization", TRUE,
  8,  "Insight \u2192 Theme\nMapping",   "Organization", TRUE,
  9,  "Theme\nSummarization",           "Synthesis", TRUE,
  10, "Orphan\nDetection",              "Coverage", TRUE,
  11, "Orphan\nReinsertion",            "Synthesis", TRUE,
  12, "Final\nOutput",                  "Output", FALSE
) |>
  mutate(
    x = 0,
    y = rev(seq_len(n())),
    width = 4.4,
    height = 0.78,
    xmin = x - width / 2,
    xmax = x + width / 2,
    ymin = y - height / 2,
    ymax = y + height / 2
  )

arrows <- steps |>
  arrange(id) |>
  mutate(
    x_start = x,
    y_start = ymin,
    x_end = lead(x),
    y_end = lead(ymax)
  ) |>
  filter(!is.na(x_end))

# Loop arrow: from Orphan Reinsertion back to Theme Schema
loop_arrow <- tibble::tibble(
  x_start = 2.2,
  y_start = steps$y[steps$id == 10],
  x_mid   = 3.35,
  y_mid1  = steps$y[steps$id == 10],
  y_mid2  = steps$y[steps$id == 6],
  x_end   = 2.2,
  y_end   = steps$y[steps$id == 6]
)

loop_box <- steps |>
  filter(in_loop) |>
  summarise(
    xmin = min(xmin) - 0.35,
    xmax = max(xmax) + 0.35,
    ymin = min(ymin) - 0.25,
    ymax = max(ymax) + 0.25
  )

steps_poly <- steps %>%
  rowwise() %>%
  mutate(
    x = list(c(xmin, xmax, xmax, xmin)),
    y = list(c(ymin, ymin, ymax, ymax))
  ) %>%
  tidyr::unnest(c(x, y))

# -----------------------------
# Plot
# -----------------------------

p <- ggplot() +
  # Iteration loop box
  geom_rect(
    data = loop_box,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = NA,
    color = "grey25",
    linewidth = 0.7,
    linetype = "dashed"
  ) +
  
  # Main arrows
  geom_segment(
    data = arrows,
    aes(x = x_start, y = y_start - 0.05, xend = x_end, yend = y_end + 0.05),
    arrow = arrow(length = unit(0.18, "inches"), type = "closed"),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Loop arrow: right side path
  geom_segment(
    data = loop_arrow,
    aes(x = x_start, y = y_start, xend = x_mid, yend = y_mid1),
    linewidth = 0.55,
    color = "grey20"
  ) +
  geom_segment(
    data = loop_arrow,
    aes(x = x_mid, y = y_mid1, xend = x_mid, yend = y_mid2),
    linewidth = 0.55,
    color = "grey20"
  ) +
  geom_segment(
    data = loop_arrow,
    aes(x = x_mid, y = y_mid2, xend = x_end, yend = y_end),
    arrow = arrow(length = unit(0.18, "inches"), type = "closed"),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Boxes
  geom_shape(
    data = steps_poly,
    aes(x = x, y = y, group = id, fill = stage),
    radius = unit(0.08, "inches"),
    color = "grey15",
    linewidth = 0.65
  ) +
  
  # Labels
  geom_text(
    data = steps,
    aes(x = x, y = y, label = label),
    size = 4.4,
    fontface = "bold",
    lineheight = 0.9
  ) +
  
  # Loop label
  annotate(
    "text",
    x = -3.45,
    y = mean(c(loop_box$ymin, loop_box$ymax)),
    label = "Iterative\nrefinement\nuntil schema\nstabilizes",
    size = 3.7,
    fontface = "italic",
    hjust = 0.5,
    lineheight = 0.95
  ) +
  
  scale_fill_manual(
    values = c(
      "Framing" = "#F2F2F2",
      "Reading" = "#DCEBFA",
      "Organization" = "#E8DFF5",
      "Synthesis" = "#FCE4C9",
      "Coverage" = "#F7D6D6",
      "Output" = "#DDEEDC"
    )
  ) +
  
  coord_equal(
    xlim = c(-4.5, 4.1),
    ylim = c(min(steps$y) - 0.5, max(steps$y) + 0.5),
    expand = FALSE
  ) +
  theme_void(base_size = 13) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    plot.margin = margin(20, 20, 20, 20)
  )

p

# Save as vector PDF for LaTeX / arXiv
ggsave("readingmachine_pipeline.pdf", p, width = 7, height = 10, device = cairo_pdf)

# Optional SVG export
# install.packages("svglite")
# ggsave("readingmachine_pipeline.svg", p, width = 7, height = 10, device = svglite::svglite)