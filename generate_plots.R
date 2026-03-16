library(classInt)
library(treemap)
library(tidyverse)
library(lubridate)
library(sf)
library(maps)
library(choroplethr)
library(choroplethrMaps)
library(scales)
library(patchwork) 
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(corrplot)
library(grid)        
library(ggrepel)
library(vcd)

# Create output directory
dir.create("plots", showWarnings = FALSE)

# Load data
heat_events <- read.csv("EDAV-Project/data_clean/heat_events_clean.csv", stringsAsFactors = TRUE)
hosp <- read.csv("EDAV-Project/data_clean/hosp_clean.csv", stringsAsFactors = TRUE)
er_visits <- read.csv("EDAV-Project/data_clean/er_visits_clean.csv", stringsAsFactors = TRUE)
deaths <- read.csv("EDAV-Project/data_clean/deaths_clean.csv", stringsAsFactors = TRUE)

## Plot 1: Trends over time
outcomes_time <- bind_rows(
  hosp |> 
    group_by(Year) |> 
    summarise(value = sum(Hosps, na.rm = TRUE), outcome = "Hospitalizations"),
  er_visits |> 
    group_by(Year) |> 
    summarise(value = sum(ER.Visits, na.rm = TRUE), outcome = "ER Visits"),
  deaths |> 
    group_by(Year) |> 
    summarise(value = sum(as.numeric(Deaths), na.rm = TRUE), outcome = "Deaths")
)

outcomes_indexed <- outcomes_time |>
  group_by(outcome) |>
  mutate(baseline = value[Year == min(Year)],
         index = (value / baseline) * 100) |>
  ungroup()

p1 <- ggplot(outcomes_indexed, aes(x = Year, y = index, color = outcome)) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  geom_line(linewidth = 1.3, alpha = 0.8) +
  geom_point(size = 2.5) +
  scale_color_manual(
    values = c("Hospitalizations" = "#e31a1c", 
               "ER Visits" = "#fd8d3c", 
               "Deaths" = "#800026")
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0.05, 0.1))) +
  scale_x_continuous(breaks = seq(2000, 2022, by = 4)) +
  labs(
    title = "Heat-Related Health Impacts Have Surged Since 2000",
    subtitle = "All three outcomes indexed to baseline year (Year 2000 = 100%)",
    x = NULL,
    y = "Index (Year 2000 = 100%)",
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.justification = "left")
ggsave("plots/plot1_trends.png", p1, width = 10, height = 6)

## Plot 2: Heat events vs Hospitalizations standardized
combined <- heat_events |>
  group_by(Year) |>
  summarise(heat_events = sum(EHE, na.rm = TRUE)) |>
  left_join(
    hosp |>
      group_by(Year) |>
      summarise(hospitalizations = sum(Hosps, na.rm = TRUE)),
    by = "Year"
  )

combined_standardized <- combined |>
  mutate(
    heat_z = scale(heat_events)[,1],
    hosp_z = scale(hospitalizations)[,1]
  ) |>
  pivot_longer(cols = c(heat_z, hosp_z),
               names_to = "Metric",
               values_to = "Standardized_Value")

p2 <- ggplot(combined_standardized, aes(x = Year, y = Standardized_Value, color = Metric)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  scale_color_manual(
    values = c("heat_z" = "firebrick", "hosp_z" = "royalblue"),
    labels = c("Heat Events", "Hospitalizations")
  ) +
  labs(
    title = "Trends in Heat Events and Hospitalizations",
    x = "Year",
    y = "Standard Deviations from Mean",
    color = ""
  ) +
  theme_minimal(base_size = 12) + theme(legend.position = "top")
ggsave("plots/plot2_standardized.png", p2, width = 10, height = 6)

## Plot 3: Heatmap
top_10_states <- heat_events |>
  group_by(State) |>
  summarise(total = sum(EHE)) |>
  arrange(desc(total)) |> head(10) |> pull(State)

heatmap_data <- heat_events |>
  group_by(State, Year) |>
  summarise(heat_events = sum(EHE), .groups = "drop") |>
  filter(State %in% top_10_states)

state_order <- heatmap_data |>
  group_by(State) |>
  summarise(total = sum(heat_events)) |>
  arrange(total) |> pull(State)

heatmap_data$State <- factor(heatmap_data$State, levels = state_order)

p3 <- ggplot(heatmap_data, aes(x = Year, y = State, fill = heat_events)) +
  geom_tile(color = "white", linewidth = 0.8) +
  scale_fill_gradientn(
    colors = c("#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"),
    name = "Heat\nEvents",
    labels = scales::comma
  ) +
  theme_minimal(base_size = 13)
ggsave("plots/plot3_heatmap.png", p3, width = 10, height = 6)

print("R plots generated successfully")
