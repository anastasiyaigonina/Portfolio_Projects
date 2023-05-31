# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# First we load the data
df = pd.read_csv('customer_shopping_data.csv')
print(df.head())

# Check information about columns, number of null entries and datatypes
df.info()

# Invoice date has wrong data type, we will fix it
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

# Also, we create new columns for year and month that can be useful for further analysis
df['year'] = df['invoice_date'].dt.strftime("%Y")
df['month'] = df['invoice_date'].dt.strftime("%m")

print(df.head())

# Get descriptive information about dataset
print(df.describe())

# Are there any duplicated rows?
print(df.duplicated().sum())

# Add new column total money spent, which will be useful later
df['total'] = df['price'] * df['quantity']
print(df.head())

# New column with age group
age_groups = [18, 24, 34, 44, 54, 64, 70]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-70']
df['age_group'] = pd.cut(df['age'], bins=age_groups, labels=labels)
age_cats = pd.CategoricalDtype(['18-24', '25-34', '35-44', '45-54', '55-64', '65-70'], ordered=True)
df['age_group'] = df['age_group'].astype(age_cats)
print(df.head())

# First, investigate gender columns and see if we get some insights
sns.countplot(data=df, x='gender').set(title='Gender and number of transactions')
plt.show()

# Build histogram of age distribution per number of transactions
sns.histplot(data=df, x='age_group').set(title='Age distribution and number of transactions')
plt.show()

# Which age group spent more money?
age_group_total = df.groupby('age_group')['total'].sum().reset_index()
sns.barplot(data=age_group_total, x='age_group', y='total', palette='Paired'). \
    set(title='Age group and total spent')
plt.show()

# Check age distribution among males and females
sns.boxplot(data=df, x='gender', y='age', palette='Set2')
plt.title('Age distribution for females and males customers')
plt.show()

# Explore payment methods
sns.countplot(x='payment_method', data=df).set(title='Payment method and number of transactions')
plt.show()

# Most customers are paying by cash. But what about the amount of money spent and payment method?
df_payment = pd.DataFrame(df.groupby('payment_method')['total'].sum())
print(df_payment)

# Investigate which products price customers usually prefer
df.price.hist()
plt.title('Price distribution')
plt.show()

# Discover popular categories
df_category_count = df.groupby('category')['invoice_no'].count().reset_index()
print(df_category_count.sort_values(by='invoice_no', ascending=False))

# Visualize popular categories per number of transactions and total amount spent
df_category_total = df.groupby('category')['total'].sum().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
sns.barplot(data=df_category_count, x='category', y='invoice_no', ax=ax1).set(
    title='Categories and number of transactions')
sns.barplot(data=df_category_total, x='category', y='total', ax=ax2).set(title='Categories and total amount spent')
ax1.tick_params('x', labelrotation=45)
ax2.tick_params('x', labelrotation=45)
plt.show()

# What is the average price per category?
avg_price_category = pd.DataFrame(df.groupby('category')['price'].mean().sort_values(ascending=False))
avg_price_category.columns = ['average_price']
print(avg_price_category)

# Distribution number of transaction per age groups and gender
print(pd.crosstab([df.age_group, df.gender], df.category, values=df.invoice_no, aggfunc=(['count'])))

# How much did each combination of gender and age group spent in total in different categories?
print(pd.crosstab([df.age_group, df.gender], df.category,
                  values=df.total, aggfunc=np.sum, normalize='columns').
                    applymap(lambda x: "{0:.0f}%".format(100 * x)))

# The most popular shopping malls by number of transactions
print(pd.DataFrame(df['shopping_mall'].value_counts()))

# And which shopping mall has the biggest amount of money spent? Let us visualize it
malls = df.groupby('shopping_mall')['total'].sum().reset_index().sort_values(by='total', ascending=False)
plt.figure(figsize=(10, 6))
squarify.plot(sizes=malls['total'], label=malls['shopping_mall'], alpha=.8, color=sns.color_palette("magma",
                                                                                                    len(malls)),
              ec='black')
plt.title('Most popular shopping malls by amount of money spent')
plt.axis('off')
plt.show()

# Interesting to see how the total amount of money spent changed per year
# But we will not include data from March 2023 since it is not full (only till the 8th of March available)
year_month = df[df['invoice_date'] < '2023-03-01'].groupby(['year', 'month'])['total'].sum()
year_month.plot(grid=True)
plt.title('Total spent per month in 2021-2023', size=14)
plt.tick_params('x', labelrotation=45)
plt.show()

# Choose relevant columns for predictions
df_model = df[['gender', 'age_group', 'category', 'price', 'total']]

# Get dummy data
df_dum = pd.get_dummies(df_model)
print(df_dum.head())

# Train test split
X = df_dum.drop('total', axis=1)
y = df_dum.total.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Decision tree
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=6, max_leaf_nodes=10)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

# Neural network
nn = MLPRegressor(hidden_layer_sizes=(50, 50))
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)

nn_mae = mean_absolute_error(y_test, nn_pred)
nn_mse = mean_squared_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)

# Compare performance of the models
print('Linear Regression - MAE:', lr_mae, 'MSE:', lr_mse, 'R-squared:', lr_r2)
print('Decision Tree - MAE:', dt_mae, 'MSE:', dt_mse, 'R-squared:', dt_r2)
print('Neural Network - MAE:', nn_mae, 'MSE:', nn_mse, 'R-squared:', nn_r2)

# Cross validation
print(np.mean(cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
print(np.mean(cross_val_score(dt, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
print(np.mean(cross_val_score(nn, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# Load dataset with currency rate
usd_try = pd.read_csv('USD_TRY Historical Data.csv', parse_dates=['Date'])

# Choose relevant columns
usd_try = usd_try[['Date', 'Price']]
usd_try.columns = ['date', 'rate']
print(usd_try.head())

# Visualize rate change over time
sns.lineplot(data=usd_try, x='date', y='rate')
plt.title('USD to TRY overtime')
plt.tick_params('x', labelrotation=45)
plt.show()

# Prepare customers data dataset
daily_total = pd.DataFrame(df.groupby('invoice_date')['total'].sum().reset_index())
print(daily_total.head())

# Merge datasets
merged = pd.merge(daily_total, usd_try, left_on='invoice_date', right_on='date', how='inner')
print(merged.head())
merged.info()
print(merged.describe())

# Plot relationship between total revenue and currency rate
sns.relplot(data=merged, x='date', y='total', hue='rate')
plt.title('Total revenue and currency rate for daily sales')
plt.tick_params('x', labelrotation=45)
plt.show()

# Find correlation between total revenue and rate
print(merged.corr())
