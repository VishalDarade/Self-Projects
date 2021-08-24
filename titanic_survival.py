# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 21:29:07 2021

@author: Prince
"""
## Analysis of titanic disaster to predict suevival of passengers ##

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

## Load dataset and know data ##

train = pd.read_csv('E:/ML/titanic/Train.csv')
test = pd.read_csv('E:/ML/titanic/Test.csv')

train.head(), train.shape
test.head(), test.shape

train.columns
test.columns

## Exploratory analysis - Univariate, multivariate ##

train.isnull().sum()     #Missing values
test.isna().sum()

train.describe(include='all')      #Distribution
test.describe(include= 'all')

train['Survived'].value_counts().plot.bar()
plt.xlabel('Survived', fontsize=16)
train['Survived'].value_counts(normalize=True)

pd.crosstab(train['Survived'], train['Pclass'], normalize=True)
sns.countplot('Survived', hue='Pclass', data=train)

sns.pairplot(data=train, hue='Survived')    #relation with dependant var by plot
sns.pairplot(test)

sns.heatmap(train.corr(), annot=True, cmap='Reds')    #correlation b/w variables
plt.title('Correlation in train variables')

sns.heatmap(test.corr(), annot=True)
plt.title('Correlation in test variables')

## Data Preprocessing - missing value, feature engg, normalization##

train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

sns.kdeplot(train.Age)                          #imputation
train['Age'].fillna(train['Age'].mean(), inplace=True)
train.dropna(inplace=True)

test['Age'].fillna(test['Age'].mean(), inplace=True)
sns.kdeplot(test.Age)

m = test['Fare'][test['Pclass']==3].mean()
test.fillna(m, inplace=True)

train.Sex = train.Sex.map(dict(female=1, male=0))   #feature engg
test.Sex = test.Sex.map(dict(female=1, male=0))

y_train = train['Survived']
x = train.drop('Survived', axis=1)
x_train = pd.get_dummies(x)
x_train.drop('Embarked_C', axis=1, inplace=True)

x_test = pd.get_dummies(test)
x_test.drop('Embarked_C', axis=1, inplace=True)


## Model Fitting ##

dclf = DecisionTreeClassifier(max_depth=5, random_state=0)
dclf.fit(x_train, y_train)

pred = dclf.predict(x_test)

p = pd.DataFrame()
p['PassengerId'] = range(892,1310)
p['Survived'] = pred
p.to_csv('Titanic-pred1', index=False)


#________________#_________________#________________#_________________#_________