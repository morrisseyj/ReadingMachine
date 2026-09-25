library(ggplot2)
library(ggforce)
library(dplyr)
library(tidyr)

# -----------------------------
# One-time reading pipeline
# -----------------------------

main_steps <- tibble::tribble(
  ~id, ~label,                         ~stage,         ~x,   ~y,
  1,   "Research\nQuestions",          "Framing",      -2.6, 8.50,
  2,   "Corpus\nIngestion",            "Reading",      -2.6, 7.25,
  3,   "Chunking",                     "Reading",      -2.6, 6.00,
  4,   "Insight\nExtraction",          "Reading",      -2.6, 4.75,
  5,   "Clustering",                   "Organization", -2.6, 3.50,
  6,   "Cluster\nSummarization",       "Organization", -2.6, 2.25,
  7,   "Derive Initial\nTheme Schema", "Organization", -2.6, 1.00
) |>
  mutate(
    width = 3.8,
    height = 0.90
  )

# -----------------------------
# Iterative testing cycle
# -----------------------------

cycle_steps <- tibble::tribble(
  ~id, ~label,                          ~stage,         ~x,  ~y,
  8,   "Map Insights\nto Themes",       "Organization", 1.4,  1.00,
  9,   "Populate\nThemes",              "Synthesis",    1.4, -0.25,
  10,  "Detect Omitted\nInsights",      "Coverage",     1.4, -1.50,
  11,  "Reinsert\nOrphans",             "Synthesis",    1.4, -2.75
) |>
  mutate(
    width = 3.4,
    height = 0.90
  )

# Schema revision and final output
revision_step <- tibble::tibble(
  id = 12,
  label = "Revise\nTheme Schema",
  stage = "Organization",
  x = 5.2,
  y = -0.60,
  width = 3.4,
  height = 0.90
)

final_step <- tibble::tibble(
  id = 13,
  label = "Final Cleanup\nand Output",
  stage = "Output",
  x = 5.2,
  y = -4.35,
  width = 3.4,
  height = 1.00
)

decision_step <- tibble::tibble(
  id = 14,
  label = "Theme Schema\nStable?",
  x = 5.2,
  y = -2.75,
  width = 3.4,
  height = 1.15
)

# Add node boundaries
box_steps <- bind_rows(
  main_steps,
  cycle_steps,
  revision_step,
  final_step
) |>
  mutate(
    xmin = x - width / 2,
    xmax = x + width / 2,
    ymin = y - height / 2,
    ymax = y + height / 2
  )

decision_step <- decision_step |>
  mutate(
    xmin = x - width / 2,
    xmax = x + width / 2,
    ymin = y - height / 2,
    ymax = y + height / 2
  )

# -----------------------------
# Main pipeline arrows
# -----------------------------

main_arrows <- main_steps |>
  mutate(
    ymin = y - height / 2,
    ymax = y + height / 2
  ) |>
  arrange(id) |>
  mutate(
    x_start = x,
    y_start = ymin,
    x_end = lead(x),
    y_end = lead(ymax)
  ) |>
  filter(!is.na(x_end))

# -----------------------------
# Downward arrows inside cycle
# -----------------------------

cycle_arrows <- cycle_steps |>
  mutate(
    ymin = y - height / 2,
    ymax = y + height / 2
  ) |>
  arrange(id) |>
  mutate(
    x_start = x,
    y_start = ymin,
    x_end = lead(x),
    y_end = lead(ymax)
  ) |>
  filter(!is.na(x_end))

# -----------------------------
# Initial schema -> iterative cycle
# -----------------------------

initial_schema <- box_steps |>
  filter(id == 7)

mapping_step <- box_steps |>
  filter(id == 8)

enter_cycle_arrow <- tibble::tibble(
  x_start = initial_schema$xmax,
  y_start = initial_schema$y,
  x_end = mapping_step$xmin,
  y_end = mapping_step$y
)

# -----------------------------
# Reinsertion -> stability decision
# -----------------------------

reinsertion_step <- box_steps |>
  filter(id == 11)

to_decision_arrow <- tibble::tibble(
  x_start = reinsertion_step$xmax,
  y_start = reinsertion_step$y,
  x_end = decision_step$xmin,
  y_end = decision_step$y
)

# -----------------------------
# Decision branches
# -----------------------------

revision_box <- box_steps |>
  filter(id == 12)

final_box <- box_steps |>
  filter(id == 13)

# No -> revise schema
to_revision_arrow <- tibble::tibble(
  x_start = decision_step$x,
  y_start = decision_step$ymax,
  x_end = revision_box$x,
  y_end = revision_box$ymin
)

# Yes -> final output
to_final_arrow <- tibble::tibble(
  x_start = decision_step$x,
  y_start = decision_step$ymin,
  x_end = final_box$x,
  y_end = final_box$ymax
)

# -----------------------------
# Revision -> mapping return path
# -----------------------------

return_arrow <- tibble::tibble(
  x_start = revision_box$x,
  y_start = revision_box$ymax,
  x_mid = revision_box$x,
  y_mid = mapping_step$y,
  x_end = mapping_step$xmax,
  y_end = mapping_step$y
)

# -----------------------------
# Node polygons
# -----------------------------

box_polygons <- box_steps |>
  rowwise() |>
  mutate(
    poly_x = list(c(xmin, xmax, xmax, xmin)),
    poly_y = list(c(ymin, ymin, ymax, ymax))
  ) |>
  ungroup() |>
  unnest(c(poly_x, poly_y))

decision_polygon <- decision_step |>
  rowwise() |>
  mutate(
    poly_x = list(c(
      x,
      xmax,
      x,
      xmin
    )),
    poly_y = list(c(
      ymax,
      y,
      ymin,
      y
    ))
  ) |>
  ungroup() |>
  unnest(c(poly_x, poly_y))

# -----------------------------
# Plot
# -----------------------------

p <- ggplot() +
  
  # Main reading-pipeline arrows
  geom_segment(
    data = main_arrows,
    aes(
      x = x_start,
      y = y_start - 0.04,
      xend = x_end,
      yend = y_end + 0.04
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Downward arrows in iterative cycle
  geom_segment(
    data = cycle_arrows,
    aes(
      x = x_start,
      y = y_start - 0.04,
      xend = x_end,
      yend = y_end + 0.04
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Enter iterative cycle
  geom_segment(
    data = enter_cycle_arrow,
    aes(
      x = x_start,
      y = y_start,
      xend = x_end,
      yend = y_end
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Reinsertion -> stability decision
  geom_segment(
    data = to_decision_arrow,
    aes(
      x = x_start,
      y = y_start,
      xend = x_end,
      yend = y_end
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Stability decision -> schema revision
  geom_segment(
    data = to_revision_arrow,
    aes(
      x = x_start,
      y = y_start,
      xend = x_end,
      yend = y_end
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Stability decision -> final output
  geom_segment(
    data = to_final_arrow,
    aes(
      x = x_start,
      y = y_start,
      xend = x_end,
      yend = y_end
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Revision return path: first move upward
  geom_segment(
    data = return_arrow,
    aes(
      x = x_start,
      y = y_start,
      xend = x_mid,
      yend = y_mid
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Then return left into mapping
  geom_segment(
    data = return_arrow,
    aes(
      x = x_mid,
      y = y_mid,
      xend = x_end,
      yend = y_end
    ),
    arrow = arrow(
      length = grid::unit(0.16, "inches"),
      type = "closed"
    ),
    linewidth = 0.55,
    color = "grey20"
  ) +
  
  # Rounded rectangular nodes
  geom_shape(
    data = box_polygons,
    aes(
      x = poly_x,
      y = poly_y,
      group = id,
      fill = stage
    ),
    radius = grid::unit(0.08, "inches"),
    color = "grey15",
    linewidth = 0.65
  ) +
  
  # Stability decision diamond
  geom_polygon(
    data = decision_polygon,
    aes(
      x = poly_x,
      y = poly_y,
      group = id
    ),
    fill = "#F2F2F2",
    color = "grey15",
    linewidth = 0.65
  ) +
  
  # Box labels
  geom_text(
    data = box_steps,
    aes(
      x = x,
      y = y,
      label = label
    ),
    size = 4.4,
    fontface = "bold",
    lineheight = 0.9
  ) +
  
  # Decision label
  geom_text(
    data = decision_step,
    aes(
      x = x,
      y = y,
      label = label
    ),
    size = 4.1,
    fontface = "bold",
    lineheight = 0.9
  ) +
  
  # Label the iterative part
  annotate(
    "text",
    x = 3.15,
    y = 1.75,
    label = "Iterative schema testing and refinement",
    size = 3.5,
    fontface = "italic",
    color = "grey25"
  ) +
  
  # No branch
  annotate(
    "text",
    x = 5.50,
    y = mean(c(decision_step$ymax, revision_box$ymin)),
    label = "No",
    size = 3.4,
    fontface = "italic",
    hjust = 0
  ) +
  
  # Yes branch
  annotate(
    "text",
    x = 5.50,
    y = mean(c(decision_step$ymin, final_box$ymax)),
    label = "Yes",
    size = 3.4,
    fontface = "italic",
    hjust = 0
  ) +
  
  scale_fill_manual(
    values = c(
      "Framing" = "#F2F2F2",
      "Reading" = "#DCEBFA",
      "Organization" = "#E8DFF5",
      "Synthesis" = "#FCE4C9",
      "Coverage" = "#F7D6D6",
      "Output" = "#DDEEDC"
    ),
    breaks = c(
      "Framing",
      "Reading",
      "Organization",
      "Synthesis",
      "Coverage",
      "Output"
    )
  ) +
  
  coord_cartesian(
    xlim = c(-4.7, 7.15),
    ylim = c(-5.0, 9.1),
    expand = FALSE,
    clip = "off"
  ) +
  
  theme_void(base_size = 13) +
  
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5),
    legend.key.width = grid::unit(0.40, "inches"),
    legend.spacing.x = grid::unit(0.08, "inches"),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  
  guides(
    fill = guide_legend(
      nrow = 1,
      byrow = TRUE
    )
  )

p

# Save as vector PDF for LaTeX / arXiv
ggsave(
  "C:\\Users\\jmorrissey\\Documents\\python_projects\\ReadingMachine\\commentary\\arxiv-2026\\readingmachine_pipeline.pdf",
  p,
  width = 9,
  height = 9.5,
  device = cairo_pdf
)

#Save as SVG for repo whitepaper
install.packages("svglite")
ggsave(
  "C:\\Users\\jmorrissey\\Documents\\python_projects\\ReadingMachine\\documentation\\readingmachine_pipeline.svg",
  p,
  width = 9,
  height = 9.5,
  device = svglite::svglite
)