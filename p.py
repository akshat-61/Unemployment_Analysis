# Firstly we will import sime of the importent files for the implementation 

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
data.fillna(data.mean(), inplace = True)

# Now we will use EDA which means Exploratory Data Analysis to get more insights about the dataset.

#Firstly we'll check the Unemployment rate overtime.

plt.figure(figsize=(12, 6))
sns.barplot(x = 'Region', y = 'Unemployment Rate', data=data, ci=None)
