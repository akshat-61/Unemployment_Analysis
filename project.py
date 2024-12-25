# Firstly we will import sime of the importent files for the implementation 

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# # This is used to create an animated, static and interactive visualistaion in python
import seaborn as sns
# It is built on top of matplotlib and is used to design creative attractive and informative statistical graphics

data = pd.read_csv('Oasis Infobyte.csv')
# Here we are loading the datadset into the compiler

#Now lets overview the dataset


print("Dataset Head")
print(data.head())
print("\nDataset Info: ")
data.info()
print("\nSummary Statistics: ")
print(data.describe())

# Now we will check if there consist any kind of missing values in the dataset.
print("\nMissing Values")
print(data.isnull().sum())

# Now will fill that empty values with the mean of the rest
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)


# Now we will use EDA which means Exploratory Data Analysis to get more insights about the dataset.

#Firstly we'll check the Unemployment rate overtime.

plt.figure(figsize=(10, 6))
sns.lineplot(x = 'Region', y = 'Unemployment Rate', data=data)
plt.tile('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemploymet Rate')
plt.xticks(rotation=45)
plt.show()


#Secondly we'll check the Unemployment Rate By Region

plt.figure(figsize=(12, 6))
sns.barplot(x = 'Region', y = 'Unemployment Rate', data=data, ci=None)
plt.title('Unemployment Rate By Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.show()


# Thirdly we'll check the Unemployment rate by Gender

plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='Unemployment Rate', data=data)
plt.title('Unemployment Rate by gender')
plt.xlabel('Gender')
plt.ylabel('Unemployment Rate')
plt.show()

# Now we will do Correlation Analysis

correlation = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap= 'coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Predictive Modeling

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Defining features and target
features = ['Feature1', 'Feature2', 'Feature3']  
target = 'Unemployment Rate'


#Split data into the traingin and testing set
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Now we are training the model
model = RandomForestRegressor(random_state = 42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test) 

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

region_mean = data.groupby('Region')['Unemployment Rate'].mean().reset_index()
print("\nRegion with the highest Unemployment rates: ")
print(region_mean.head())


# Now we will save the model
plt.figure(figsize=(10, 6))
plt.bar(region_mean['Region'], region_mean['Unemployment Rate'])
plt.title('Average Unemplpoymant Rate by Region')
plt.xlabel('Region')
plt.ylabel('Average Unemployment Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('unemployment_rate_by_region.png')
plt.show()