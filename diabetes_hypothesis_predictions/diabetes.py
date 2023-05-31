# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
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
# Let us iterate from those columns and build plots in one go.

col_names = ['HighChol', 'HighBP', 'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk']
a = 3  # number of rows
b = 2  # number of columns
c = 1  # plot counter

fig = plt.figure(figsize=(12, 15))
for i in col_names:
    plt.subplot(a, b, c)
    ax = sns.countplot(data=df, x=i, hue='Diabetes_binary', palette='Set2')
    ax.set(title='{}'.format(i))
    ax.set(xlabel=None)
    ax.set_xticklabels(['No', 'Yes'])
    ax.legend(['No-diabetes', 'Diabetics'])
    c = c + 1
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

# Now use the ttest since we use one numeric, one categorical variable
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

# Use ttest
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

# Since we have two categorical variables,we will use chi2 test
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

# Chi2 test for two categorical variables
chi2, p_value_4, dof, exp_freq = chi2_contingency(contingency2)
if p_value_4 < 0.05:
    print('Reject Null Hypothesis')
else:
    print('Failed to reject Null Hypothesis')

# Predictions
# Choose columns for model
df_model = df[
    ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'GenHlth', 'DiffWalk', 'Age', 'HeartDiseaseorAttack', 'PhysHlth']]

# Train test split
X = df_model.drop('Diabetes_binary', axis=1)
y = df_model['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Decision Tree
model_1 = DecisionTreeClassifier()
model_1.fit(X_train, y_train)

# Calculate model performance
predictions = model_1.predict(X_test)
model_1_score = accuracy_score(y_test, predictions)
print('Accuracy score for Decision Tree is', model_1_score)

# Random Forest
model_2 = RandomForestClassifier()
model_2.fit(X_train, y_train)

# Calculate model performance
predictions = model_2.predict(X_test)
model_2_score = accuracy_score(y_test, predictions)
print('Accuracy score for Random Forest is', model_2_score)

# XGB
model_3 = XGBClassifier()
model_3.fit(X_train, y_train)

# Calculate model performance
predictions = model_3.predict(X_test)
model_3_score = accuracy_score(y_test, predictions)
print('Accuracy score for XGB is', model_3_score)

# SVC
model_4 = SVC()
model_4.fit(X_train, y_train)

# Calculate model performance
predictions = model_4.predict(X_test)
model_4_score = accuracy_score(y_test, predictions)
print('Accuracy score for SVC is', model_4_score)

# KNeighbors
model_5 = KNeighborsClassifier()
model_5.fit(X_train, y_train)

# Calculate model performance
predictions = model_5.predict(X_test)
model_5_score = accuracy_score(y_test, predictions)
print('Accuracy score for KNeighbors is', model_5_score)

# MLP
model_6 = MLPClassifier()
model_6.fit(X_train, y_train)

# Calculate model performance
predictions = model_6.predict(X_test)
model_6_score = accuracy_score(y_test, predictions)
print('Accuracy score for MLP is', model_6_score)

# Compare models performance
Models = ['Decision Tree', 'Random Forest', 'XBG', 'SVC', 'KNeighbors', 'MLP']
Scores = [model_1_score, model_2_score, model_3_score, model_4_score, model_5_score, model_6_score]

performance = pd.DataFrame(list(zip(Models, Scores)), columns=['Models', 'Accuracy_score']).sort_values(
    'Accuracy_score',
    ascending=False)
print(performance)
