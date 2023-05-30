# Load library
library(tidyverse)
library(dplyr)
library(gplots)
library(ggplot2)

# Load data
df <- read_csv('Downloads/insurance.csv')
head(df)

# Change data type
df$sex <- as.factor(df$sex)
df$smoker <- as.factor(df$smoker)
df$region <- as.factor(df$region)
str(df)

# Get summary
summary(df)

# Is any missing values?
sum(is.na(df))

# Lets explore distribution of age in this dataset
hist(df$age, 
     main = "Distribution of Age",
     xlab = "Age",
     ylab = "Count",
     col = "skyblue",
     border = "white")

# Check sex distribution 
sex_counts <- table(df$sex)
colors <- c("#FF9999", "#66B2FF") 

pie(sex_counts, 
    main = "Sex Distribution",
    col = colors,
    labels = paste0(names(sex_counts), " (", sex_counts, ")"),
    cex = 1.2,
    radius = 1.2)

# Explore BMI
hist(df$bmi, 
     main = "Distribution of BMI",
     xlab = "BMI",
     ylab = "Count",
     col = "darkgreen",
     border = "white")

# Explore charges
hist(df$charges,
     main = "Distribution of Charges",
     xlab = "Charges",
     ylab = "Frequency",
     col = "#009999")

# Check distribution of charges by sex
boxplot(charges ~ sex, data = df,
        main = "Distribution of Charges by Sex",
        xlab = "Sex",
        ylab = "Charges",
        col = c("pink", "skyblue"))

# Average charges by sex
df %>% 
  group_by(sex) %>% 
  summarize(average_charges = mean(charges))

# Check distribution of charges by smoker
boxplot(charges ~ smoker, data = df,
        main = "Distribution of Charges by Smoker",
        xlab = "Smoker",
        ylab = "Charges",
        col = c("darkred", "darkgreen"))

# Average charges by smoker
df %>% 
  group_by(smoker) %>% 
  summarize(average_charges = mean(charges))

# Check distribution of charges by number of children
boxplot(charges ~ children, data = df,
        main = "Distribution of Charges by Number of children",
        xlab = "Number of children",
        ylab = "Charges",
        col = c("skyblue", "lightgreen", "orange", "pink", "yellow", 'purple'))

# Average charges by number of children
df %>% 
  group_by(children) %>% 
  summarize(average_charges = mean(charges)) %>%
  arrange(desc(average_charges))

# Distribution of charges by region
boxplot(charges ~ region, data = df,
        main = "Distribution of Charges by Region",
        xlab = "Region",
        ylab = "Charges",
        col = c("skyblue", "lightgreen", "orange", "pink"))

# Average charges by region
df %>% 
  group_by(region) %>% 
  summarize(average_charges = mean(charges)) %>%
  arrange(desc(average_charges))

# Relationship between age and charges
plot(df$age, df$charges,
     main = "Charges vs Age",
     xlab = "Age",
     ylab = "Charges",
     col = "skyblue",
     pch = 16)

# Lets see relationships beetween BMI and charges
plot(df$bmi, df$charges,
     main = "Charges vs BMI",
     xlab = "BMI",
     ylab = "Charges",
     col = "#009999",
     pch = 16)

# Seems like obesity affect charges a lot. Lets create new column to investigate it
df$obese <- ifelse(df$bmi >= 30, "yes", "no")
df$obese <- as.factor(df$obese)
head(df)

# Visualize obesity and charges
boxplot(charges ~ obese, data = df,
        main = "Distribution of Charges by Obesity",
        xlab = "Obese",
        ylab = "Charges",
        col = c("#0072B2", "#E69F00"))

# Average charges by obesity
df %>% 
  group_by(obese) %>% 
  summarize(average_charges = mean(charges))

# Create numeric dataframe
numeric_df <- df
numeric_df$sex <- as.numeric(numeric_df$sex)
numeric_df$smoker <- as.numeric(numeric_df$smoker)
numeric_df$region <- as.numeric(numeric_df$region)
numeric_df$obese <- as.numeric(numeric_df$obese)

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_df)
correlation_matrix

# Visualize the correlation
heatmap.2(correlation_matrix, 
          main = "Correlation Heatmap",
          col = colorRampPalette(c("white", "darkblue"))(100),
          key = FALSE,    
          symkey = FALSE,
          trace="none",
          cexCol = 1.3,    
          cexRow = 1.3,    
          srtCol = 45,      
          cellnote = round(correlation_matrix, 2),   
          notecol = "white",    
          notecex = 1)   

# As we can see smoking and obesity status significantly affect charges. Lets visualize charges by smoker and obesity
ggplot(df, aes(x = smoker, y = charges, fill = obese)) +
  geom_boxplot() +
  facet_grid(. ~ obese, labeller = labeller(obese = c("no" = "Non-Obese", "yes" = "Obese"))) +
  labs(title = "Charges by Smoker and Obese Status", x = "Smoker", y = "Charges") +
  scale_fill_manual(values = c("no" = "#66B2FF", "yes" = "#FF9999"), labels = c("Non-Obese", "Obese")) +
  theme_minimal()

# Visualize relationship between charges, age and smoking status
ggplot(df, aes(x = age, y = charges, color = smoker)) +
  geom_point() +
  labs(title = "Charges by Smoking Status and Age", x = "Age", y = "Charges", color = "Smoker") +
  theme_minimal()


# Splitting the data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(df), 0.8 * nrow(df))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Multiple linear regression to predict charges using all variables from dataset
model1 <- lm(charges ~ sex + age + children + bmi + obese + smoker + region, data = train_data)
summary(model1)

# Lets remove variables that do not affect charges much and add interaction between smoking status and obesity since as we saw before, they have the biggest impact on charges
model2 <- lm(charges ~ age + bmi + obese*smoker, data = train_data)
summary(model2)

# Predicting charges on the test data
predictions_model1<- predict(model1, newdata = test_data)
predictions_model2 <- predict(model2, newdata = test_data)

# Evaluate the models
# RMSE
mse_model1 <- mean((test_data$charges - predictions_model1)^2)
rmse_model1 <- sqrt(mse_model1)

mse_model2 <- mean((test_data$charges - predictions_model2)^2)
rmse_model2 <- sqrt(mse_model2)

rmse_model1
rmse_model2

# Create a scatter plot of predicted charges vs. actual charges
ggplot(test_data, aes(x = charges, y = predictions_model1)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual Charges", y = "Predicted Charges",
       title = "Model 1: Predicted Charges vs. Actual Charges") +
  theme_minimal()

# Assess the residuals' distribution
residuals_model1 <- test_data$charges - predictions_model1
ggplot(data.frame(residuals = residuals_model1), aes(x = residuals)) +
  geom_histogram(binwidth = 1000, fill = "skyblue", color = "black") +
  labs(x = "Residuals", y = "Frequency",
       title = "Model 1: Residuals Distribution") +
  theme_minimal()

# Create a scatter plot of predicted charges vs. actual charges
ggplot(test_data, aes(x = charges, y = predictions_model2)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual Charges", y = "Predicted Charges",
       title = "Model 2: Predicted Charges vs. Actual Charges") +
  theme_minimal()

# Assess the residuals' distribution
residuals_model2 <- test_data$charges - predictions_model2
ggplot(data.frame(residuals = residuals_model2), aes(x = residuals)) +
  geom_histogram(binwidth = 1000, fill = "skyblue", color = "black") +
  labs(x = "Residuals", y = "Frequency",
       title = "Model 2: Residuals Distribution") +
  theme_minimal()