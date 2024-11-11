# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
print(df.head())

# Check info about dataset
df.info()

# Might be useful later to divide dataset into two - diabetes and no_diabetes
df_no = df[df['Diabetes_binary'] == 0]
df_yes = df[df['Diabetes_binary'] == 1]

# Lets discover gender distribution
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
ax1 = sns.countplot(data=df_no, x='Sex', ax=ax1, palette='husl')
ax1.set(title='Gender distribution for no-diabetes')
ax1.set_xticklabels(['Female', 'Male'])

ax2 = sns.countplot(data=df_yes, x='Sex', ax=ax2, palette='husl')
ax2.set(title='Gender distribution for diabetics')
ax2.set_xticklabels(['Female', 'Male'])
plt.show()

# Check age distribution for people with diabetes
ax = sns.countplot(data=df_yes, x='Age')
ax.set(title='Age distribution for diabetics')
ax.set_xticklabels(
    ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>80'],
    rotation=45)
plt.show()

# Check BMI for people with diabetes.
# We will remove outliers for better visualization, less than 15 and greater than 60.
sns.histplot(data=df_yes, x='BMI').set(title='BMI distribution for diabetics')
plt.xlim(15, 60)
plt.show()

# Compare BMI for people with and without diabetes
ax = sns.boxplot(data=df, x='Diabetes_binary', y='BMI', palette='Paired')
ax.set(title='BMI distribution for no-diabetes and diabetics')
ax.set_xticklabels(['No diabetes', 'Diabetic'])
plt.ylim(15, 60)
plt.show()

# There are some binary columns that we can visually compare data between no-diabetes and diabetics.
# Lets iterate from those columns and build plots in one go.

col_names = ['HighChol', 'HighBP', 'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk']
a = 3  # number of rows
b = 2  # number of columns
c = 1  # plot counter

fig = plt.figure(figsize=(12, 15))
for col in col_names:
    plt.subplot(a, b, c)
    ax = sns.countplot(data=df, x='Diabetes_binary', hue=col, palette='Set2')
    ax.set(title=f'Count of Diabetes Status by {col}', xlabel='Diabetes Status', ylabel='Count')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.legend(title=col)
    c += 1  
plt.tight_layout()
plt.show()

# Create a correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Visualize relationship between all variables
plt.figure(figsize=(12, 10))
sns.heatmap(data=corr_matrix, cmap='crest')
plt.show()

# Hypothesis testing
# 1. Do no-diabetes and diabetics have the same BMI?
# H0 - no-diabetes and diabetics have the same average BMI.
# Ha - no-diabetes and diabetics have different average BMI.

# Let us prepare dataset
df_no_bmi = df_no['BMI']
df_yes_bmi = df_yes['BMI']

# Check the average BMI
print('Average BMI for diabetics is {} and no-diabetes is {} '.format(df_yes_bmi.mean(), df_no_bmi.mean()))

# Check visually how BMI distribution looks like
sns.kdeplot(df_yes_bmi, color='red')
sns.kdeplot(df_no_bmi, color='green')
plt.grid()
plt.title('BMI distribution')
plt.legend(['Diabetics', 'No-diabetes'])
plt.show()

# Now use the ttest since we compare the means of two independent groups
ttest, p_value_1 = stats.ttest_ind(df_yes_bmi, df_no_bmi)
if p_value_1 < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Failed to reject Null Hypothesis')

# Do no-diabetes and diabetics have the same number of poor physical health days per month?
# H0 - No-diabetes and diabetics have the same average number of poor physical health days per month.
# Ha - No-diabetes and diabetics have the different average number of poor physical health days per month

# Prepare datasets
df_no_ph = df_no['PhysHlth']
df_yes_ph = df_yes['PhysHlth']

# Check the distribution of average number of poor physical health days per month
sns.kdeplot(df_yes_ph, color='red')
sns.kdeplot(df_no_ph, color='green')
plt.grid()
plt.title('Distribution of average number of poor physical health days per month')
plt.legend(['Diabetes', 'No-diabetes'])
plt.show()

# Count average days
print('Average days of poor physical health for diabetics is {} and no-diabetics is {} '.format(df_yes_ph.mean(),
                                                                                                df_no_ph.mean()))

# Use t-test to compare the means of two independent groups
ttest, p_value_2 = stats.ttest_ind(df_yes_ph, df_no_ph)
if p_value_2 < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Failed to reject Null Hypothesis')

# Is the proportion of high cholesterol significantly different across diabetics and no-diabetes?¶
# H0 - High cholesterol proportion is not significantly different across diabetics and no-diabetes.
# Ha - High cholesterol proportion is different across diabetics and no-diabetes.

# Prepare data
contingency = pd.crosstab(df.Diabetes_binary, df.HighChol)
print(contingency)

# Visualize high cholesterol proportions
ax = contingency.plot(kind='bar')
ax.set(xlabel=None)
ax.set_xticklabels(['No-diabetes', 'Diabetics'], rotation=20)
ax.legend(['No High Cholesterol', 'High Cholesterol'])
plt.title('High cholesterol proportion across diabetics and no-diabetes')
plt.show()

# Since we are trying to determine whether there is a significant association 
# between two categorical variables,we will use chi2 test
chi2, p_value_3, dof, exp_freq = chi2_contingency(contingency)
if p_value_3 < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Failed to reject Null Hypothesis')

# Is the proportion of high blood pressure significantly different across diabetics and no-diabetes?
# H0 - High blood pressure proportion is not significantly different across diabetics and no-diabetes.
# Ha - High blood pressure proportion is different across diabetics and no-diabetes.

# Prepare data
contingency2 = pd.crosstab(df.Diabetes_binary, df.HighBP)
print(contingency2)

# Visualize proportion of high blood pressure for people with and without diabetes
ax = contingency2.plot(kind='bar')
ax.set(xlabel=None)
ax.set_xticklabels(['No-diabetes', 'Diabetics'], rotation=20)
ax.legend(['No High BP', 'High BP'])
plt.title('High blood pressure proportion across diabetics and no-diabetes')
plt.show()

# Chi2 test for testing relationships between two categorical variables
chi2, p_value_4, dof, exp_freq = chi2_contingency(contingency2)
if p_value_4 < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Failed to reject Null Hypothesis')

# Predictions
# Feature selection 

# Define features and target
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2 
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
X = df.drop('Diabetes_binary', axis=1) 
y = df['Diabetes_binary']

# Feature selection using SelectKBest with Chi-Square
selector = SelectKBest(score_func=chi2, k=12)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_columns = selector.get_support(indices=True)
important_features = X.columns[selected_columns].tolist()

# Display the selected features
print(important_features)

# Create a DataFrame for the selected features
X_selected = pd.DataFrame(X_new, columns=important_features)

# Handling class imbalance

# Handling class imbalance using NearMiss to undersample the majority class
nm = NearMiss(version = 1 , n_neighbors = 10)
X_sm,y_sm= nm.fit_resample(X_selected,y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

# Scaling 

# Standardizing the features (important for models sensitive to feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling

# Lets start with Logistic Regression
# Creating and training the logistic regression model
logistic_model = LogisticRegression(random_state=42, max_iter=200)

# Train model using scaled data
logistic_model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Evaluating the model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
class_report_logistic = classification_report(y_test, y_pred_logistic)

# Output the evaluation results
print(f"Accuracy of Logistic Regression Model: {accuracy_logistic:.3f}")
print("Classification Report:")
print(class_report_logistic)

# K-Neighbors
# Initialize the KNN classifier with a specified number of neighbors (k)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model on scaled data to improve performance
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_knn = knn_model.predict(X_test_scaled)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)

# Output the evaluation results
print(f"KNN Accuracy: {accuracy_knn:.3f}")
print("Classification Report:")
print(class_report_knn)

# Decision Tree
# Initialize the Decision Tree classifier
decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth= 12)

# Train the model
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate the model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
class_report_tree = classification_report(y_test, y_pred_tree)

# Output the evaluation results
print(f"Decision Tree Accuracy: {accuracy_tree:.3f}")
print("Classification Report:")
print(class_report_tree)

# Random Forest
# Building the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)

# Train the model 
random_forest_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred_random_forest = random_forest_model.predict(X_test)

# Evaluating the model
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
class_report_random_forest = classification_report(y_test, y_pred_random_forest)

# Output the evaluation results
print(f"Accuracy for Random Forest: {accuracy_random_forest:.3f}")
print("Classification Report:")
print(class_report_random_forest)

# Support Vector Machine
# Building the SVM model
svm_model = SVC(random_state=42, C=1.0)  

# Train using scaled data
svm_model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluating the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)

# Output the evaluation results
print(f"Accuracy of SVM Model: {accuracy_svm:.3f}")
print("Classification Report:")
print(class_report_svm)

# Multi-Layer Perceptron
# Initialize the MLP classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

# Train the model using scaled data
mlp_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Evaluate the model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
class_report_mlp = classification_report(y_test, y_pred_mlp)

# Output the evaluation results
print(f"MLP Accuracy: {accuracy_mlp:.3f}")
print("Classification Report:")
print(class_report_mlp)

# Extreme Gradient Boosting
# Initialize the XGBoost classifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

# Output the evaluation results
print(f"XGBoost Accuracy: {accuracy_xgb:.3f}")
print("Classification Report:")
print(class_report_xgb)

# Hyperparameter Tuning for XGBoost
# Define the parameter grid to search 
param_grid = {
    'n_estimators': [50, 100, 150],  
    'learning_rate': [0.01, 0.1, 0.2, 0.3], 
    'max_depth': [3, 4, 5, 6, 7],  
    'gamma': [0, 0.1, 0.2],  
    'min_child_weight': [1, 2, 3]  
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=0
)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions with the best model on the test set
y_pred_best = best_model.predict(X_test)

# Evaluate the model
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)

# Output the evaluation results
print(f"Best Parameters: {best_params}")
print(f"Accuracy with Best Parameters: {accuracy_best:.3f}")
print("Classification Report:")
print(class_report_best)

# Compare models performance
Models = [
    'Logistic Regression', 
    'KNN', 
    'Decision Tree', 
    'Random Forest', 
    'SVM', 
    'MLP', 
    'XBG', 
    'XGBoost (Tuned)'
]

Scores = [
    accuracy_logistic, 
    accuracy_knn, 
    accuracy_tree, 
    accuracy_random_forest, 
    accuracy_svm, 
    accuracy_mlp, 
    accuracy_xgb, 
    accuracy_best
]

# Create a DataFrame for better visual comparison
performance = pd.DataFrame(
    list(zip(Models, Scores)), 
    columns = ['Models', 'Accuracy_score']
).sort_values('Accuracy_score', ascending=False)

# Round the accuracy scores for better readability
performance['Accuracy_score'] = performance['Accuracy_score'].round(3)

# Display the performance DataFrame
print(performance)