##### Script adding further variables of interest to ACLED #####
## Variables are: Pre-post invasion dummy, Federal Election month, 
## Police violence during protest
## Some further data cleaning is applied

# 
# pacman::p_load(rio, tidyverse, readr, broom, 
#                lubridate, gt, gtsummary, survey, readxl,
#                gridExtra, knitr, haven, stargazer, nnet, ggeffects)

#### Load Data if not loaded
# acled <- import("data/acled_with_org_indicator.csv") 

#### Get unique observations ####

length(unique(acled$notes))
# Finding duplicate observations in 'variable'
duplicates <- duplicated(acled$notes)

# Getting the row IDs of these duplicates
row_ids <- which(duplicates)

# Print events
acled$notes[row_ids]

# Remove duplicates 
acled_clean <- acled[!duplicated(acled$event_id_cnty), ]

min(acled_clean$event_date)
max(acled_clean$event_date)

#### General Data Cleaning ####

# Convert date column to Date object and create DVs:
# pre/post invasion
# election_month
acled_clean <- acled_clean %>% 
  mutate(date = ymd(event_date),
         pred_labels = recode(pred_labels,
                              "war (pro)" = "war_pro",
                              "war (anti)" = "war_anti"),
         month_year = format(date, "%Y-%m"),
         post_invasion = factor(case_when(
           date >= as.Date("2022-02-24") ~ 1,
           TRUE ~ 0)),
         police_violence = case_when(
           sub_event_type %in% c("Protest with intervention", "Excessive force against protesters") ~ 1,
           TRUE ~ 0
         ),
         election_month = factor(case_when(
           month_year %in% c("2021-09", "2018-03", "2024-03", "2020-06", "2020-07") ~ 1,
           TRUE ~ 0)),
         # protest_invasion: Pension reform (July-Sept 2018), Const. ref (June-July 2020), Ukr invasion (Feb-March 2022)
         protest_invasion = factor(case_when(
           month_year %in% c("2018-07", "2018-08", "2018-09", 
                             "2020-06", "2020-07", 
                             "2022-02", "2022-03") ~ 1,
           TRUE ~ 0))
  ) |> 
  filter(year(date) <= 2024)  # Exclude jan 2025, so we have full years

# recode region names
mapping <- c(
  'Saint Petersburg' = 'The City of Saint-Petersburg',
  'Sverdlovsk' = 'Sverdlovsk region',
  'Moscow' = 'The City of Moscow',
  'Republic of Tatarstan' = 'Republic of Tatarstan',
  'Novosibirsk' = 'Novosibirsk region',
  'Novgorod' = 'Novgorod region',
  'Samara' = 'Samara region',
  'Republic of Mari El' = 'Republic of Mariy El',
  'Republic of Dagestan' = 'Republic of Daghestan',
  'Tomsk' = 'Tomsk region',
  'Republic of Karachay-Cherkessia' = 'Karachaev-Circassian Republic',
  'Moscow Oblast' = 'Moscow region',
  'Volgograd' = 'Volgograd region',
  'Chelyabinsk' = 'Chelyabinsk region',
  'Kaliningrad' = 'Kaliningrad region',
  'Khabarovsk' = 'Khabarovsk territory',
  'Republic of Khakassia' = 'Republic of Khakassia',
  'Perm' = 'Perm region',
  'Vologda' = 'Vologda region',
  'Republic of Sakha' = 'Republic of Sakha (Yakutia)',
  'Tula' = 'Tula region',
  'Kemerovo' = 'Kemerovo region',
  'Altai' = 'Altai territory',
  'Republic of Bashkortostan' = 'Republic of Bashkortastan',
  'Penza' = 'Penza region',
  'Republic of Chuvash' = 'Chuvash Republic',
  'Krasnoyarsk' = 'Krasnoyarsk territory',
  'Orenburg' = 'Orenburg region',
  'Tyumen' = 'Tyumen region',
  'Republic of Kabardino-Balkaria' = 'Kabardian-Balkar Republic',
  'Kostroma' = 'Kostroma region',
  'Vladimir' = 'Vladimir region',
  'Arkhangelsk' = 'Arkhangelsk region',
  'Rostov' = 'Rostov region',
  'Krasnodar' = 'Krasnodar territory',
  'Khanty-Mansi' = 'Khanty-Mansi autonomous',
  'Kaluga' = 'Kaluga region',
  'Republic of Karelia' = 'Republic of Karelia',
  'Voronezh' = 'Voronezh region',
  'Ryazan' = 'Ryazan region',
  'Zabaykalskiy' = 'Chita region',
  'Ulyanovsk' = 'Ulyanovsk region',
  'Republic of Komi' = 'Republic of Komi',
  'Republic of Buryatia' = 'Republic of Buryatia',
  'Nizhny Novgorod' = 'Nizhny novgorod region',
  'Ivanovo' = 'Ivanovo region',
  'Saratov' = 'Saratov region',
  'Irkutsk' = 'Irkutsk region',
  'Republic of Altai' = 'Republic of Altai',
  'Primorskiy' = 'Primorsky territory',
  'Republic of Chechnya' = 'Chechen Republic',
  'Belgorod' = 'Belgorod region',
  'Lipetsk' = 'Lipetsk region',
  'Udmurt Republic' = 'Udmurt Republic',
  'Astrakhan' = 'Astrakhan region',
  'Kirov' = 'Kirov region',
  'Yaroslavl' = 'Yaroslavl region',
  'Kursk' = 'Kursk region',
  'Omsk' = 'Omsk region',
  'Republic of Mordovia' = 'Republic of Mordovia',
  'Tambov' = 'Tambov region',
  'Smolensk' = 'Smolensk region',
  'Republic of Ingushetia' = 'Republic of Ingushetia',
  'Oryol' = 'Oryol region',
  'Pskov' = 'Pskov region',
  'Republic of Tuva' = 'Republic of Tyva',
  'Republic of North Ossetia-Alania' = 'Republic of North Ossetia -Alania',
  'Jewish Autonomous Oblast' = 'Jewish autonomous region',
  'Kamchatka' = 'Kamchatka region',
  'Magadan' = 'Magadan region',
  'Republic of Kalmykia' = 'Republic of Kalmykia',
  'Yamalo-Nenets' = 'Yamalo-Nenets autonomous',
  'Bryansk' = 'Bryansk region',
  'Stavropol' = 'Stavropol territory',
  'Tver' = 'Tver region',
  'Murmansk' = 'Murmansk region',
  'Republic of Adygea' = 'Republic of Adygeya',
  'Amur' = 'Amur region',
  'Sakhalin' = 'Sakhalin region',
  'Leningrad' = 'Leningrad region',
  'Chukotka' = 'Chukotka autonomous area',
  'Nenets' = 'Nenets autonomous area',
  'Kurgan' = 'Kurgan region'
)


# apply mapping
acled_clean$federal_subject <- mapping[acled_clean$admin1]

acled_clean <- acled_clean %>%
  left_join(codes, by = c("federal_subject" = "region_name_en"))


acled_clean <- acled_clean %>% 
  mutate(source_origin = case_when(str_detect(source_scale, "International") ~ "international",
                                   TRUE ~ "russian"))


#### Recoding Variables for Analyses ####

## create pol expand and topic relevel ##
acled_clean = acled_clean %>% 
  mutate(pol_expand = case_when(pred_labels %in% c("political", "war_anti", "war_pro") ~ 1,
                                T ~ 0),
         pol_expand2 = case_when(pred_labels %in% c("political", "war_anti") ~ 1,
                                 T ~ 0)) %>% 
  mutate(topics_recoded = relevel(as.factor(case_when(pol_expand == 1 ~ "0",
                                                      pred_labels %in% c("cultural", "legal") ~ "1",
                                                      pred_labels == "environmental" ~ "2",
                                                      pred_labels %in% c("economic", "social") ~ "3")),
                                  ref="0"),
         topics_recoded2 = relevel(as.factor(case_when(pol_expand == 1 ~ "0",
                                                       pred_labels %in% c("cultural", "legal") ~ "1",
                                                       pred_labels == "environmental" ~ "2",
                                                       pred_labels == "economic" ~ "3",
                                                       pred_labels == "social" ~ "4",
         )),
         ref="0"),
         topics_recoded3 = relevel(as.factor(case_when(pol_expand == 1 ~ "0",
                                                       pred_labels %in% c("cultural", "legal", "environmental") ~ "1",
                                                       pred_labels  %in% c("economic", "social") ~ "2"
         )),
         ref="0"),
  ) %>% 
  mutate(pro_kremlin_indicator = case_when(org_indicator == "pro_kremlin" ~ 1,
                                           T ~ 0))


##### (un)authorised indicator #####

#dictionary
dict = quanteda::dictionary(list(unauthorized = c("not authorized", "unauthorized", "unlawful",
                                                  "unsanctioned", "did not approve", 
                                                  "did not authorize", "not sanctioned",
                                                  "not been authorized"),
                                 authorized = c("was authorized", "an authorized", 
                                                "was approved", "was sanctioned", 
                                                "authorized protest*")))

# tokens with no pre-processing 
corp_acled = corpus(acled, text_field = "notes",
                    docid_field = "event_id_cnty")

acled_toks = tokens(corp_acled)

dict_toks = tokens_lookup(acled_toks, dictionary = dict)

res = convert(dfm(dict_toks), to = "data.frame") %>% 
  rename(event_id_cnty = doc_id)

# merge 
acled_clean = acled_clean %>% left_join(res, by = "event_id_cnty") 




## authorised and unauthorised dummies ##
acled_clean <- acled_clean %>%
  mutate(
    auth_rec = case_when(
      authorized == 0 ~ 0,
      authorized %in% 1:2 ~ 1,
      TRUE ~ NA_real_
    ),
    unauth_rec = case_when(
      unauthorized == 0 ~ 0,
      unauthorized %in% 1:2 ~ 1,
      TRUE ~ NA_real_
    )
  )

## organiser ##
acled_clean = acled_clean %>% 
  mutate(org_c1 = case_when(org_indicator=="other" ~ 1,
                            T ~ 0)
  )


## time indicator ##
acled_clean <- acled_clean %>%
  mutate(
    time_indicator = case_when(
      # Group 0: Dates up to February 23, 2022
      date %in% as.Date(c("2022-02-14", "2022-02-15", "2022-02-16", "2022-02-17",
                          "2022-02-18", "2022-02-19", "2022-02-20", "2022-02-21",
                          "2022-02-22", "2022-02-23")) ~ 0,
      
      # Group 1: Dates from February 24 to March 3, 2022
      date %in% as.Date(c("2022-02-24", "2022-02-25", "2022-02-26", "2022-02-27",
                          "2022-02-28", "2022-03-01", "2022-03-02", "2022-03-03")) ~ 1,
      
      # Group 2: Dates from March 4 to March 20, 2022
      date %in% as.Date(c("2022-03-04", "2022-03-05", "2022-03-06", "2022-03-07",
                          "2022-03-08", "2022-03-09", "2022-03-10", "2022-03-11",
                          "2022-03-12", "2022-03-13", "2022-03-14", "2022-03-15",
                          "2022-03-16", "2022-03-17", "2022-03-18", "2022-03-19",
                          "2022-03-20")) ~ 2,
      
      # Default case (optional)
      TRUE ~ NA_real_
    )
  )


#### Save Cleaned Data ####
write_excel_csv(acled_clean, "data/acled_with_dvs_and_controls_16_07_2024_utf8byte.csv")
