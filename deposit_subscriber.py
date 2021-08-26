# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:59:11 2021

@author: Prince
"""
## Prediction of whether client will subcribe to credit or not ##

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


## Data Extraction ##
train = pd.read_csv('E:/ML/projects/credit_subscriber/train.csv')
test = pd.read_csv('E:/ML/projects/credit_subscriber/test.csv')

train.shape, test.shape


## Exploratory analysis(EDA)- variables, univariate/bivariate analysis,
#                            missing value, outliers ##

train.dtypes
train.describe()
train.describe(include=object)

train.isna().sum()                     #missing value detection
test.isnull().sum()

plt.plot(train['age'])                 #univariate analysis
sns.distplot(train['age'])
train.job.value_counts().plot.bar()
train['marital'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
plt.hist(train['balance'], bins=200)
sns.kdeplot(train.duration)
train.age.plot.box( patch_artist=True)         #outlier detection
train.balance.plot.box(vert=False)
train.duration.plot.box()
train.plot.box()
                                               #bivariate analysis
pd.crosstab(train['job'], train.subscribed).plot.bar(stacked=True)
marr = pd.crosstab(train.marital, train['subscribed'])
marr.div(marr.sum(1).astype(float), axis=0).plot.bar(stacked=True)
sns.countplot('subscribed', hue='education', data=train)
sns.kdeplot(x= train.duration, y=train.day)
plt.scatter(train.day, train.duration)
plt.scatter(train.job, train.age)
pd.crosstab(train.contact, train.subscribed)
                                                 #multivariate analysis
sns.pairplot(data=train,  hue='subscribed')
sns.heatmap(train.corr(), cmap='RdYlBu', annot=True)
sns.clustermap(train.corr(), cmap='RdYlBu', annot=True, center=0)


## Data tranformation/munging ##

train.drop_duplicates(inplace=True)         #remove duplicates

y = train['subscribed']                   #variable transformation
y = y.map(dict(yes=1, no=0))                 
x = train.drop('subscribed', axis=1)

y_train = y
x_train = pd.get_dummies(x)
x_test = pd.get_dummies(test)



## Model building ##

logreg = LogisticRegression()         #Accuracy = 0.893
logreg.fit(x_train, y_train)
pred1 = logreg.predict(x_test)

clf = DecisionTreeClassifier(max_depth=6, random_state=0) #Accuracy = 0.906
clf.fit(x_train, y_train)
pred2 = clf.predict(x_test)

p = pd.DataFrame()                       #results presentation
p['pred1'] = pred1
p['pred1'].replace(1,'yes', inplace = True)
p['pred1'].replace(0,'no', inplace = True)

p['pred2'] = pred2
p.pred2.replace(1,'yes', inplace = True)
p.pred2.replace(0,'no', inplace = True)

p.to_excel('pred_sub.xlsx')


