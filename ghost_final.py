# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:28:05 2018

@author: ASHIK
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = train.append(test,ignore_index='True')
id = test.id

data.drop(['id'],1,inplace=True)

encoded_color = pd.get_dummies(data.color)

data_dummy = pd.concat([data,encoded_color],axis=1)
data_dummy.drop(['color'],1,inplace=True)

X = data_dummy.drop('type',1).values 
y = train.type.values

X_train = X[0:len(train),:]
X_test = X[len(train):len(data),:]
le = LabelEncoder()
y_train = le.fit_transform(y)
clf = RandomForestClassifier(n_estimators=300,n_jobs=5)

clf.fit(X_train,y_train)
pred = clf.predict(X_test)
pred1 = le.inverse_transform(pred)
print(pred1)


sub = pd.DataFrame({'id':id , 'type':pred1})
sub.to_csv("C:/Users/ASHIK/Desktop/Projects/Kaggle_competetion_GhostsPrediction/ghost_submission.csv", index=False) 






