import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

data = pd.read_csv('Oasis Infobyte.csv')

print("Dataset Head")
print(data.head())
print("\nDataset Info: ")
data.info()
print("\nSummary Statistics: ")
print(data.describe())

print("\nMissing Values")
print(data.isnull().sum())

data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

plt.figure(figsize=(10, 6))
sns.lineplot(x = 'Region', y = 'Unemployment Rate', data=data)
plt.tile('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemploymet Rate')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x = 'Region', y = 'Unemployment Rate', data=data, ci=None)
plt.title('Unemployment Rate By Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='Unemployment Rate', data=data)
plt.title('Unemployment Rate by gender')
plt.xlabel('Gender')
plt.ylabel('Unemployment Rate')
plt.show()

correlation = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap= 'coolwarm')
plt.title('Correlation Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = ['Feature1', 'Feature2', 'Feature3']  
target = 'Unemployment Rate'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

region_mean = data.groupby('Region')['Unemployment Rate'].mean().reset_index()
print("\nRegion with the highest Unemployment rates: ")
print(region_mean.head())


plt.figure(figsize=(10, 6))
plt.bar(region_mean['Region'], region_mean['Unemployment Rate'])
plt.title('Average Unemplpoymant Rate by Region')
plt.xlabel('Region')
plt.ylabel('Average Unemployment Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('unemployment_rate_by_region.png')
plt.show()