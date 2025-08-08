# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline

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
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
# Feature engineering
# BMI Categories
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


df['BMI_cat'] = df['BMI'].apply(bmi_category)

# BMI x Age (Age is coded 1-13, higher = older age group)
df['BMI_Age_interaction'] = df['BMI'] * df['Age']

# High_risk group: Obese + Age 10+ (65+ per BRFSS age codes)
df['HighRisk_Obese_Old'] = ((df['BMI_cat'] == 'Obese') & (df['Age'] >= 10)).astype(int)

# Convert BMI_cat to ordinal codes 
df['BMI_cat_code'] = pd.Categorical(df['BMI_cat'], 
                                    categories=['Underweight','Normal','Overweight','Obese'],
                                    ordered=True).codes
df = df.drop(columns=['BMI_cat'])

# Train-test split

# Define features and target
X = df.drop('Diabetes_binary', axis=1) 
y = df['Diabetes_binary']

# Splitting the data into training and testing sets, while preserving class distribution using stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing and feature selection
# Specify the feature types for preprocessing
numeric_features = ['BMI', 'BMI_Age_interaction', 'MentHlth', 'PhysHlth']
categorical_features = ['Age','Education', 'Income', 'GenHlth', 'BMI_cat_code']


# Build a preprocessing pipeline using ColumnTransformer to handle numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OrdinalEncoder(), categorical_features),
    ],
    remainder='passthrough'
)

# Feature selection using ANOVA F-test (f_classif) to select the top 16 features
feature_selector = SelectKBest(score_func=f_classif, k=16)

# Training and evaluation
# Class weights to address class imbalance 
# Define models with class_weight
weighted_models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=200, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=12, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced'),
    "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced', n_estimators=250, verbose=-1)
}

# List to store results
results = []

for model_name, base_model in weighted_models.items():
        steps = [
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selector),
            ('classifier', base_model)
        ]

        # Build pipeline
        clf = ImbPipeline(steps=steps)

        # Fit
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, -1] if hasattr(clf.named_steps['classifier'], "predict_proba") else None

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan

        # Store results
        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "AUC Score": round(auc, 3),
        })
    
# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Sampling techniques to handle class imbalance

# Define sampling techniques
sampling_methods = {
    "RandomOverSampler": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "EditedNN": EditedNearestNeighbours(n_neighbors=3),
    "TomekLinks": TomekLinks(),
}

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=12),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
    "LGBM": LGBMClassifier(random_state=42, n_estimators=250, verbose=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators = 250),
    "XGB": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
} 

# List to store results
results2 = []

# Iterate through all combinations of sampling techniques and classification models
for method_name, sampler in sampling_methods.items():
    for model_name, base_model in models.items():
        steps = [
            ('preprocessor', preprocessor),
            ('sampler', sampler),
            ('feature_selection', feature_selector),
            ('classifier', base_model)
        ]

        # Build pipeline
        clf = ImbPipeline(steps=steps)

        # Fit
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, -1] if hasattr(clf.named_steps['classifier'], "predict_proba") else None

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan

        # Store results
        results2.append({
            "Sampling Method": method_name,
            "Model": model_name,
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "AUC Score": round(auc, 3),
        })


# Convert results to DataFrame
results_df2 = pd.DataFrame(results2)
print(results_df2)
