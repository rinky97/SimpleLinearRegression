# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:27:23 2017

@author: Rinky
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting simple linear model onto training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set salaries
y_pred = regressor.predict(X_test)

#graphs to display the actual salary points and simple linear regression line
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()