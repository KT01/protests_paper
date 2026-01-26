library(tidyverse)

# Old original data
acled_ru_2018_2023_mz <- read_csv(
  "data/acled_ru_2018_2023_mz.csv",
  col_types = cols(admin3 = col_character())
) |>
  # 14 November 2023
  mutate(event_date = dmy(event_date)) |>
  # Contains problematic values and is not used
  select(-civilian_targeting)


# New orignal data
acled_ru_2020_2024 <- readxl::read_excel(
  "data/ACLED_Russia_Protest_2020_2024.xlsx",
  .name_repair = tolower
) |>
  mutate(
    event_date = as_date(event_date),
    time_precision = as.double(time_precision),
    admin3 = as.character(admin3)
  ) |>
  select(-civilian_targeting)

# 3752 matched keys
acled_merged <-
  acled_ru_2018_2023_mz |>
  rows_upsert(
    acled_ru_2020_2024,
    by = c("event_id_cnty")
  )

# Save pre-merged data
write_csv(
    acled_merged,
    "data/acled_merged.csv"
)
