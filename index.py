# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:17:15 2020

@author: FUJITSU
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


titanicdata = pd.read_csv('D:\\Koding\\Python\\titanic\\train.csv')
titanicdata.head(10)

#coba = pd.DataFrame(titanicdata)

#ngecek banyaknya jumlah data
#print(str(len(titanicdata.index)))

#cek data penumpang selamat

#sns.countplot(x = "Survived", data = titanicdata)
#sns.countplot(x = "Survived", hue="Sex", data = titanicdata)
#sns.countplot(x = "Survived", hue="Pclass", data = titanicdata)

#sns.countplot(x = "Survived", hue="Age", data = titanicdata)
#titanicdata["Age"].plot.hist()
#titanicdata["Fare"].plot.hist(bins = 20, figsize =(10, 5))
#titanicdata.info()
#sns.countplot(x = "SibSp", data = titanicdata)

#DATA WRANGLING/CLEANING

#titanicdata.isnull()
#titanicdata.isnull().sum()
#sns.heatmap(titanicdata.isnull(), "yticklabels"==False, cmap = "viridis")
#sns.boxplot(x = "Pclass", y = "Age", data = titanicdata)
#titanicdata.drop("Cabin", axis = 1, inplace = True) #kalau axis = 0 untuk index, 1 untuk kolom, True langsung saja sedangkan false ditampilkan
#titanicdata.dropna(inplace=True)
#sns.heatmap(titanicdata.isnull(), yticlabels = False, cbar = False)
#titanicdata.isnull().sum()
#titanicdata.head(2)
#titanicdata.info()
#titanicdata.drop("Cabin", axis = 1, inplace = True)
#titanicdata.info()
#titanicdata.head(2)
#len(titanicdata)
titanicdata.drop("Cabin", axis = 1, inplace = True)
titanicdata.dropna(inplace=True)
titanicdata.head(2)
#len(titanicdata)
#titanicdata.info()

sex = pd.get_dummies(titanicdata['Sex'], drop_first=True)
sex.head(5)
embark = pd.get_dummies(titanicdata['Embarked'], drop_first=True)
embark.head(5)
#Pclass
pcl = pd.get_dummies(titanicdata['Pclass'], drop_first=True)
pcl.head(5)
titanicdata = pd.concat([titanicdata, sex, embark, pcl], axis = 1)
titanicdata.head(5)
titanicdata.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
titanicdata.drop(['PassengerId'], axis = 1, inplace = True)
titanicdata.head()
titanicdata.drop(['Pclass'], axis = 1, inplace = True)
titanicdata.head()

x = titanicdata.drop("Survived", axis = 1)
y = titanicdata["Survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)
from sklearn.metrics import classification_report

classification_report(y_test, predictions)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)
