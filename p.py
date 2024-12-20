# Firstly we will import sime of the importent files for the implementation 

import csv
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # This is used to create an animated, static and interactive visualistaion in python
# import seaborn as sns
# It is built on top of matplotlib and is used to design creative attractive and informative statistical graphics

data = pd.read.csv('C:\Users\Public')
# Here we are loading the datadset into the compiler

#Now lets overview the dataset

print("Dataset Head")
print(data.head())
print("\nDataset Info: ")
data.info()
print("\nSummary Statistics: ")
print(data.describe())



