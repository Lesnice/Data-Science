#!/usr/bin/env python
# coding: utf-8

# # Regression modelling using insurance Dataset

# In[1]:


#Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #for feature engineering 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error # for␣evaluating ml models


# In[2]:


df = pd.read_csv(r'C:\Users\ka416\OneDrive - University of Exeter\Documents\data\ds-ml-main\ds-ml-main\datasets\insurance_costs\insurance.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.nunique() #checking for any duplicates using unique values


# In[6]:


df.duplicated().value_counts()


# In[7]:


# droping duplicate values
df.drop_duplicates(inplace=True)


# In[8]:


df.shape


# # #Exploratory Data analysis

# In[9]:


# Visualize box plots for numerical columns 
df.boxplot(column=['age', 'bmi', 'children', 'charges']) 
plt.title('Box plot of Numerical Columns')
plt.show()


# In[10]:


df.boxplot(column=[ 'bmi'] )
plt.show()


# In[11]:


df.boxplot(column=[ 'charges'] )
plt.show()


# In[12]:


df_clean = df[(df['charges'] < 20000) & (df['bmi'] < 45)] #removing outliers


# In[13]:


df_clean.boxplot(column=['age', 'bmi', 'children', 'charges'])
plt.title('Box plot of columns')
plt.show()


# In[14]:


df_clean.describe()


# In[15]:


smoker_dist = df_clean['smoker'].value_counts()
plt.pie(smoker_dist, labels=smoker_dist.index, autopct = '%1.1f%%')
plt.title('Smoker Distribution')
plt.show()


# In[16]:


smokersByRegion = df_clean.groupby('region')['smoker'].value_counts().unstack().fillna(0)
smokersByRegion.plot(kind = 'bar')
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('smokers by Region')
plt.legend(title='Smoker', loc='upper right')
plt.show()


# In[17]:


chargesByRegion = df_clean.groupby('region')['charges'].sum()
plt.bar(chargesByRegion.index, chargesByRegion.values, color='green')
plt.xlabel('Region')
plt.ylabel('Total charges')
plt.title('Total Charges by Region')
plt.xticks(rotation = 45)
plt.show()


# In[18]:


gendercounts = df_clean['sex'].value_counts() 
plt.pie(gendercounts, labels=gendercounts.index, autopct='%2.2f%%') 
plt.title('Distribution by sex')
plt.show()


# In[19]:


sns.violinplot(x=df_clean['children'], y=df_clean['charges'])

plt.xlabel('Number of Children')
plt.ylabel('Charges')

plt.title('Charges by Number of Children')
plt.show()


# In[20]:


# Categorize age based on the specified ranges
df_clean.loc[df_clean['age']<=19, 'age_group'] = 'teenage'
df_clean.loc[df_clean['age'].between(20,24), 'age_group'] = 'yadult'
df_clean.loc[df_clean['age'].between(25,39), 'age_group'] = 'adult'
df_clean.loc[df_clean['age']>39, 'age_group'] = 'older_adult'


# In[21]:


df_clean


# In[24]:


# Create a box plot visualizing BMI across age details and colored by heart disease status
import plotly.express as px
fig = px.box(df_clean,
             x="age_group",
             y="bmi",
             color="sex",
             title="BMI Distribution across Age Groups based on sex",
             )

# Update layout to make it more readable
fig.update_layout(xaxis_title="Age Group",
                  yaxis_title="BMI",
                  legend_title="sex")

# Display the plot
fig.show()


# In[25]:


#Distribution of BMI

sns.displot(df_clean['bmi'], color="dodgerblue", label="Compact", bins = 30, kde = True)
plt.title('BMI distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()


# In[26]:


df_clean.plot(kind='scatter', x='age', y='charges', color='gold')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Relationship between Age and Charges')
plt.show()


# In[27]:


sns.violinplot(x=df_clean['smoker'], y=df_clean['charges'])
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.title('Charges Distribution for Smokers and Non-Smokers')
plt.show()


# In[28]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats as ss


# In[31]:


ss.ks_2samp(df_clean['smoker'], df_clean['sex'])


# In[34]:


df_clean['age_group2'] = pd.cut(df_clean['age'], bins=[0, 25, 40, 60, df_clean['age'].max()], labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
df_clean.sample(5)


# In[35]:


df_encoded = pd.get_dummies(df_clean, columns=['region'], prefix='region', dtype=int)
df_encoded.sample(5)


# In[36]:


label_encoder = LabelEncoder()
df_encoded['smoker_encoded'] = label_encoder.fit_transform(df_encoded['smoker'])
df_encoded.sample(5)


# In[37]:


df_encoded['sex_encoded'] = label_encoder.fit_transform(df_encoded['sex'])
df_encoded.sample(5)


# In[38]:


df_encoded = df_encoded[[x for x in df_encoded.columns if x not in ['smoker', 'sex']]]
df_encoded.sample(5)


# In[43]:


num_cols = [x for x in df_encoded.columns if x not in ['age_group2']]
corr_matrix = df_encoded[num_cols].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="magma") 
plt.title("Correlation Matrix")
plt.show()

# Identify relevant features based on correlation
threshold = 0.3
relevant_features = corr_matrix[(corr_matrix['charges'].abs() > threshold) & (corr_matrix.index != 'charges')].index.tolist() 
print("Relevant features based on correlation:") 
print(relevant_features)


# In[44]:


# Select the relevant features
X = df_encoded[['age', 'smoker_encoded']]
y = df_encoded['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[47]:


# Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)

# Plot actual vs. predicted values for Decision Tree
plt.figure(figsize=(8, 4))
plt.scatter(y_test, dt_predictions, color='blue', label='Decision Tree')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--')
plt.title('Decision Tree: Actual vs. Predicted') 
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# Plot actual vs. predicted values for Random Forest
plt.figure(figsize=(8, 4))
plt.scatter(y_test, rf_predictions, color='magenta', label='Random Forest')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='pink', linestyle='--')
plt.title('Random Forest: Actual vs. Predicted') 
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)

# Plot actual vs. predicted values for Gradient Boosting
plt.figure(figsize=(8, 4))
plt.scatter(y_test, gb_predictions, color='black', label='Gradient Boosting')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Gradient Boosting: Actual vs. Predicted') 
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()


# In[48]:


# Print the evaluation metrics
print("Decision Tree - MSE: ", dt_mse)
print("Decision Tree - MAE: ", dt_mae)
print("Random Forest - MSE: ", rf_mse)
print("Random Forest - MAE: ", rf_mae)
print("Gradient Boosting - MSE: ", gb_mse)
print("Gradient Boosting - MAE: ", gb_mae)


# In[ ]:


#feature importance


# In[49]:


# Decision Tree
print("Decision Tree:")

# Feature importances
importance = dt_model.feature_importances_ 

for i, feature in enumerate(X.columns):
    print(f"{feature}: {importance[i]}") 

print()

# Random Forest
print("Random Forest:")

# Feature importances
importance = rf_model.feature_importances_

for i, feature in enumerate(X.columns): 
    print(f"{feature}: {importance[i]}")

print()

# Gradient Boosting
print("Gradient Boosting:")

# Feature importances
importance = gb_model.feature_importances_ 
for i, feature in enumerate(X.columns):
    print(f"{feature}: {importance[i]}")


# In[ ]:


#inference on sample data


# In[50]:


# Example input for prediction
new_data = pd.DataFrame({'age': [30], 'smoker_encoded': [1]})

# Decision Tree
dt_predictions = dt_model.predict(new_data)
print("Decision Tree Predictions:", dt_predictions)

# Random Forest
rf_predictions = rf_model.predict(new_data)
print("Random Forest Predictions:", rf_predictions)

# Gradient Boosting
gb_predictions = gb_model.predict(new_data)
print("Gradient Boosting Predictions:", gb_predictions)


# In[51]:


# Example input for prediction
new_data = pd.DataFrame({'age': [35], 'smoker_encoded': [0]}) # Gradient Boosting
gb_predictions = gb_model.predict(new_data)
print("Gradient Boosting Predictions:", gb_predictions)


# In[52]:


# Example input for prediction
new_data = pd.DataFrame({'age': [35], 'smoker_encoded': [1]}) # Gradient Boosting
gb_predictions = gb_model.predict(new_data)
print("Gradient Boosting Predictions:", gb_predictions)


# In[53]:


# Example input for prediction
new_data = pd.DataFrame({'age': [67], 'smoker_encoded': [0]}) # Gradient Boosting
gb_predictions = gb_model.predict(new_data)
print("Gradient Boosting Predictions:", gb_predictions)


# In[54]:


# Example input for prediction
new_data = pd.DataFrame({'age': [67], 'smoker_encoded': [1]}) # Gradient Boosting
gb_predictions = gb_model.predict(new_data)
print("Gradient Boosting Predictions:", gb_predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




