#######################################
# Visualisation script for "Differentiated Repression: Evidence from Russian Protests" paper
# for Crowd size reporting patterns and Source of reporting patterns
#######################################


# pacman::p_load(tidyverse, rio, ggplot2,
#                sandwich, lmtest, lubridate, margins, stargazer,
#                sjPlot, interplot, ggeffects)


#### Data Cleaning to Aid Visualisations ####

acled_filtered_for_viz = acled_clean %>% 
  mutate(crowd_size = str_remove(tags, "crowd size=")) %>% 
  mutate(crowd_size = case_when(crowd_size == "no report" ~ NA,
                                TRUE ~ crowd_size))

acled_filtered_for_viz <- acled_filtered_for_viz %>%
  mutate(date = as.Date(event_date)) %>%  
  filter(date >= as.Date("2021-01-01") & date <= as.Date("2024-12-31")) %>%
  mutate(month = floor_date(date, unit = "month"))


##### Crowd Sizes #####

acled_filtered_for_viz %>% group_by(post_invasion) %>% 
  summarise(nas = sum(is.na(crowd_size)),
            ns = n())

# t-test
acled_filtered_for_viz <- acled_filtered_for_viz %>%
  mutate(crowd_size_missing = ifelse(is.na(crowd_size), 1, 0))

t_test_result <- t.test(crowd_size_missing ~ post_invasion, data = acled_filtered_for_viz)

na_shares <- acled_filtered_for_viz %>%
  group_by(month) %>%
  summarise(
    total_count = n(),
    not_na_count = sum(!is.na(crowd_size)),
    na_count = sum(is.na(crowd_size))
  ) %>%
  mutate(share_reported = (not_na_count / total_count)*100)


na_share_plot <- ggplot(na_shares, aes(x = month, y = share_reported)) +
  geom_line(linetype = "solid") +
  geom_point() +
  scale_x_date(
    date_breaks = "1 month",                       # Tick every month
    labels = function(x) format(x, "%b %y"),       # Format as 'Jan 21', 'Feb 21', etc.
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(0, 100),                            # Manually set Y-axis range
    breaks = seq(0, 100, by = 10)                  # Y-axis ticks every 10 units
  ) +
  labs(title = "",
       x = "",
       y = "Shate of Protest Reports") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),  # Add a black box
    panel.background = element_blank(),            # Keep background clean
    plot.background = element_blank()              # Keep outer plot background clean
  ) +
  geom_vline(xintercept = as.Date("2022-02-24"), linetype = "solid", color = "black") +
  annotate(geom = "text", x = as.Date("2022-02-24"), y = 65, label = "Invasion", 
           vjust = -0.5, hjust = 0.5, color = "black", angle = 90, size = 3) +
  geom_vline(xintercept = as.Date("2022-09-21"), linetype = "solid", color = "black") +
  annotate(geom = "text", x = as.Date("2022-09-21"), y = 65, label = "Mobilisation", 
           vjust = -0.5, hjust = 0.5, color = "black", angle = 90, size = 3)

na_share_plot
ggsave("plots/share_crowd_sizes.png", plot = na_share_plot, width = 10, height = 6, dpi = 300)


##### sources #### 

acled_source <- acled_filtered_for_viz %>%
  separate_rows(source_scale, sep = "-") %>% 
  mutate(source_origin = case_when(source_scale == "International" ~ "International",
                                   # source_scale == "New media" ~ "New media",
                                   TRUE ~ "Domestic"))
# t-test
acled_source <- acled_source %>%
  mutate(rely_on_domestic = ifelse(source_origin=="Domestic", 1, 0))

t_test_result <- t.test(rely_on_domestic ~ post_invasion, data = acled_source)

# plot
sources_share <- acled_source %>%
  group_by(month, source_origin) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(month) %>%
  mutate(share = count / sum(count))

source_share_plot <- ggplot(sources_share, aes(x = month, y = share, linetype = source_origin)) +
  geom_line() +
  geom_point() +
  scale_x_date(
    date_breaks = "1 month",                       # Tick every month
    labels = function(x) format(x, "%b %y"),       # Format as 'Jan 21', 'Feb 21', etc.
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(0, 1),                              # Set Y-axis range for proportions
    breaks = seq(0, 1, by = 0.1),                  # Ticks every 0.1
    labels = scales::percent_format(accuracy = 1)  # Display Y-axis as percentages
  ) +
  
  labs(
    title = "",
    x = "",
    y = "Share of Protest Records",
    linetype = "Source"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),  # Box around the plot
    panel.background = element_blank(),            # Clean background
    plot.background = element_blank()              # Clean outer background
  ) +
  geom_vline(xintercept = as.Date("2022-02-24"), linetype = "solid", color = "black") +
  annotate(
    geom = "text", x = as.Date("2022-02-24"), y = 0.6, label = "Invasion",
    vjust = -0.5, hjust = 0.5, color = "black", angle = 90, size = 3
  ) +
  geom_vline(xintercept = as.Date("2022-09-21"), linetype = "solid", color = "black") +
  annotate(
    geom = "text", x = as.Date("2022-09-21"), y = 0.6, label = "Mobilisation",
    vjust = -0.5, hjust = 0.5, color = "black", angle = 90, size = 3
  )

source_share_plot
ggsave("plots/share_info_sources.png", plot = source_share_plot, width = 10, height = 6, dpi = 300)