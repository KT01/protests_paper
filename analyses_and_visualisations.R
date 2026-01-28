#######################################
# Data analyses script for "Differentiated Repression: Evidence from Russian Protests" paper
#######################################

#### Load Data if not loaded
# acled_clean <- read_csv("data/acled_with_dvs_and_controls_16_07_2024_utf8byte.csv")

# Check if there are duplicates
duplicates <- acled_clean %>%
  group_by(event_id_cnty) %>%
  filter(n() > 1)

if (nrow(duplicates) > 0) {
  print("Duplicate rows found:")
  print(duplicates)
} else {
  print("No duplicate rows found.")
}


#### Main Model (Model 1) ####

filtered_data <- acled_clean %>% filter(pro_kremlin_indicator == 0)
data_no_moscow <- filtered_data %>% filter(federal_subject != "The City of Moscow")

model_1 <- lm(police_violence ~ factor(pol_expand) * factor(year) + auth_rec + org_c1 + 
                   election_month + factor(reg_code), data = filtered_data)

# Calculate robust standard errors
robust_se <- vcovHC(model_1, type = "HC1")
# Extract coefficients and robust standard errors
robust_results <- coeftest(model_1, vcov = robust_se)
robust_coef <- robust_results[, 1]  # Coefficients
robust_se_values <- robust_results[, 2]  # Robust standard errors

## save table ##
stargazer(
  model_1,
  type = "text",
  se = list(robust_se_values), # Replace standard errors with robust SEs
  omit = c("factor\\(reg_code\\)"), # Omit factor variables
  omit.labels = c("Federal Region FE"),              
  keep.stat = c("n", "rsq", "adj.rsq"),               
  dep.var.labels = "Police Violence",
  covariate.labels = c("Political Protest", "Year 2019", "Year 2020", "Year 2021","Year 2022","Year 2023", "Year 2024",
                       "Protest authorised", "Organisers", "Election Month", "Political x 2019", 
                       "Political x 2020", "Political x 2021", "Political x 2022", "Political x 2023", "Political x 2024"),
  
  out="tables/police_violence_main.tex"
)


## make visualisation ##
plot_1 = plot_model(model_1, type = "pred", terms = c("year", "pol_expand"
),
vcov.fun = robust_se) + theme_bw()+
  scale_color_manual(values = c("darkgrey", "black"),
                     labels = c("Non-Political", "Political"),
                     name = "")+
  xlab("")+
  ylab("Predicted Probability")+ ggtitle(NULL)

ggsave("plots/polit_protest_years.png",
       plot = plot_1, width = 10, height = 6, dpi = 300)


#### Robustness for Main Model ####

### mian model different political specification (no pro-war) ###
data_no_pro_war <- filtered_data %>% 
  filter(pred_labels != "war_pro")

model_2 <- lm(police_violence ~ factor(pol_expand) * factor(year) + auth_rec + org_c1 + 
                election_month + factor(reg_code), data = data_no_pro_war)

# Calculate robust standard errors
robust_se_2 <- vcovHC(model_2, type = "HC1")
# Extract coefficients and robust standard errors
robust_results_2 <- coeftest(model_2, vcov = robust_se)
robust_coef_2 <- robust_results_2[, 1]  # Coefficients
robust_se_values_2 <- robust_results_2[, 2]  # Robust standard errors

### logit (mian model) for robustness ###
model_1_logit <- glm(police_violence ~ factor(pol_expand) * factor(year) + auth_rec + org_c1 + 
                          election_month + factor(reg_code), 
                        data = filtered_data, 
                        family = binomial(link = "logit"))

# Calculate robust standard errors
robust_se_logit <- vcovHC(model_1_logit, type = "HC1")  # Robust covariance matrix
robust_results_logits <- coeftest(model_1_logit, vcov = robust_se_logit)  # Coefficients with robust SE
robust_coef_logits <- robust_results_logits[, 1]  # Coefficients
robust_se_values_logits <- robust_results_logits[, 2]  # Robust standard errors

### main model but no Moscow ###
model_1_no_moscow <- lm(police_violence ~ factor(pol_expand) * factor(year) + auth_rec + org_c1 + 
                             election_month + factor(reg_code), data = data_no_moscow)

# Calculate robust standard errors
robust_se_no_moscow <- vcovHC(model_1_no_moscow, type = "HC1")
# Extract coefficients and robust standard errors
robust_results_no_moscow <- coeftest(model_1_no_moscow, vcov = robust_se_no_moscow)
robust_coef_no_moscow <- robust_results_no_moscow[, 1]  # Coefficients
robust_se_values_no_moscow <- robust_results_no_moscow[, 2]  # Robust standard errors

### save table with results ###

stargazer(model_2,
          model_1_logit,
          model_1_no_moscow,
          #type = "text",
          se = list(robust_se_values_2, robust_se_values_logits, robust_se_values_no_moscow), # Replace standard errors with robust SEs
          omit = c("factor\\(reg_code\\)"), # Omit factor variables
          omit.labels = c("Federal Region FE"),              # Optional: Custom labels for omitted variables
          keep.stat = c("n", "aic", "ll"),                   # For logistic models, show sample size, AIC, log-likelihood
          dep.var.labels = "Police Violence (Logit)",
          covariate.labels = c("Political Protest", "Year 2019", "Year 2020", "Year 2021","Year 2022","Year 2023", "Year 2024",
                               "Protest authorised", "Organisers", "Election Month", "Political x 2019", 
                               "Political x 2020", "Political x 2021", "Political x 2022", "Political x 2023", "Political x 2024"),
          out="tables/police_violence_robustness.tex"
)




## plot no Moscow ##
# plot moscow
plot_m = plot_model(model_1_no_moscow, type = "pred", terms = c("year", "pol_expand"
),
vcov.fun = robust_se_values_no_moscow) + theme_bw()+
  scale_color_manual(values = c("darkgrey", "black"),
                     labels = c("Non-Political", "Political"),
                     name = "")+
  xlab("")+
  ylab("Predicted Probability")+ ggtitle(NULL)

ggsave("plots/plot_no_moscow.png",
       plot = plot_m, width = 10, height = 6, dpi = 300)



#### 2021-2022 Subset #####

subset_2021_2022 = filtered_data %>% filter(year %in% c("2021", "2022")) %>% 
  mutate(
    month_year_date = as.Date(paste0(month_year, "-01"), format = "%Y-%m-%d"),
    
    # Calculate `mycnt` as months since January 2021
    mycnt = 12 * (year(month_year_date) - 2021) + month(month_year_date),
    month_year_categorical = format(month_year_date, "%b %y")
  ) %>% 
  mutate(
    # Ensure the categorical variable is ordered
    month_year_categorical = factor(
      month_year_categorical,
      levels = format(seq(as.Date("2021-01-01"), as.Date("2022-12-01"), by = "month"), "%b %y"),
      ordered = TRUE)
  )


model_subset_21_22 <- lm(police_violence ~  pol_expand * month_year_categorical + auth_rec + org_c1 + factor(reg_code), data = subset_2021_2022)
# Calculate robust standard errors
robust_se <- vcovHC(model_subset_21_22, type = "HC1")
robust_results <- coeftest(model_subset_21_22, vcov = robust_se)
robust_coef <- robust_results[, 1]  # Coefficients
robust_se_values <- robust_results[, 2]  # Robust standard errors

stargazer(
  model_subset_21_22,
  #type = "text",
  se = list(robust_se_values), # Replace standard errors with robust SEs
  omit = c("factor\\(reg_code\\)", "pol_expand:month", "month_year_"), # Omit factor variables
  omit.labels = c("Region Code", "Month Year", "Month Year x Political"),              # Optional: Custom labels for omitted variables
  keep.stat = c("n", "rsq", "adj.rsq"),                
  dep.var.labels = "Police Violence",
  covariate.labels = c("Political", "Protest Authorised", "Organizers"), # Rename predictors
  out="tables/police_violence_subset_21_22.tex"
)

## plot
plot_subset_21_22 = plot_model(model_subset_21_22, type = "pred", terms = c("month_year_categorical", "pol_expand"
),
vcov.fun = robust_se) + theme_bw()+
  scale_color_manual(values = c("darkgrey", "black"),
                     labels = c("Non-Political", "Political"),
                     name = "")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  xlab("")+
  ylab("Predicted Probability")+ ggtitle(NULL)
plot_subset_21_22

ggsave("plots/polit_protest_2021_2022.png",
       plot = plot_subset_21_22, width = 10, height = 6, dpi = 300)
