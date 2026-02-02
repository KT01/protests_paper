# ==================================================
# Classification performance of original models
# trained/run by Maksim
# ==================================================
# Table A4: Classification Performance on the Test Set
# Label Precision Recall F1-Score
# cultural 0.91 1.00 0.95
# economic 0.85 0.96 0.90
# environmental 0.93 0.95 0.94
# legal 0.98 0.93 0.96
# political 0.98 0.96 0.97
# social 0.91 0.90 0.91
# war (anti) 0.98 0.98 0.98
# war (pro) 0.99 1.00 0.99
# Overall (Micro Avg) 0.96 0.96 0.96
# Overall (Macro Avg) 0.94 0.96 0.95

# ==================================================
# Classification performance on updated code
# run by me
# ==================================================
# Precision, Recall, and F1-Score for each label
# Label Precision Recall F1-Score
# cultural 0.91 1.00 0.95  ~ Same
# economic 0.78 0.93 0.85  ~ Worse
# environmental 0.95 0.93 0.94  ~ Better
# legal 0.96 0.93 0.95  ~ Worse
# political 0.99 0.96 0.97  ~ Better
# social 0.88 0.90 0.89  ~ Worse
# war (anti) 0.98 0.98 0.98  ~ Same
# war (pro) 0.99 1.00 0.99  ~ Same
# Overall (Micro Avg) 0.95 0.95 0.95  ~ Worse
# Overall (Macro Avg) 0.93 0.95 0.94  ~ Worse

library(tidyverse)

old_results <- read_csv("data/acled_deberta_preds_17_06_2024.csv")
new_results <- read_csv("data/acled_deberta_preds_26_01_2026.csv") |>
    select(-`...1`)

# I removed `civilian_targeting` earlier, since we don't use it
setdiff(colnames(old_results), colnames(new_results))

# Focus on only the rows that were in the old results
new_results_subset <-
    new_results |>
    filter(event_id_cnty %in% old_results$event_id_cnty) |>
    select(event_id_cnty, pred_labels, topic_manual)

# 8164
nrow(old_results)
nrow(new_results_subset)

# Differences in predicted labels
# # A tibble: 264 × 4
#    event_id_cnty pred_labels_old pred_labels_new topic_manual
#    <chr>         <chr>           <chr>           <chr>
#  1 RUS12717      environmental   economic        NA
#  2 RUS12482      political       social          NA
#  3 RUS12327      environmental   legal           NA
#  4 RUS11926      economic        political       NA
#  5 RUS10636      social          legal           NA
#  6 RUS10744      economic        social          NA
#  7 RUS10627      social          economic        NA
#  8 RUS10433      political       social          NA
#  9 RUS10266      social          environmental   NA
# 10 RUS9717       social          environmental   environmental
# # ℹ 254 more rows
# # ℹ Use `print(n = ...)` to see more rows
mismatched_results <-
    old_results |>
    select(event_id_cnty, pred_labels) |>
    left_join(
        new_results_subset,
        by = "event_id_cnty",
        suffix = c("_old", "_new")
    ) |>
    filter(
        pred_labels_old != pred_labels_new
    )

mismatched_results |>
    filter(
        topic_manual == pred_labels_new
    ) |>
    nrow() # 24 ('better')

mismatched_results |>
    filter(
        topic_manual == pred_labels_old
    ) |>
    nrow() # 33 ('worse')

# 14 rows have different notes
# Some just changed punctuation, a few semantically meaningful changes
changed_notes <-
    old_results |>
    select(event_id_cnty, notes) |>
    left_join(
        new_results |>
            filter(event_id_cnty %in% old_results$event_id_cnty) |>
            select(event_id_cnty, notes),
        by = "event_id_cnty",
        suffix = c("_old", "_new")
    ) |>
    # Ensure differences aren't just due to capitalization or extra spaces
    mutate(
        across(contains("notes"), ~ str_squish(str_to_lower(.x)))
    ) |>
    filter(notes_old != notes_new)

# ...but this does not explain any of the label changes
mismatched_results |>
    filter(event_id_cnty %in% changed_notes$event_id_cnty) |>
    nrow() # 0
