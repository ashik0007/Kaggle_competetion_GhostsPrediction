# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:12:46 2018

@author: ASHIK
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')
copy = pd.read_csv('train.csv')
data.drop(['id','type'],1,inplace=True)

le = LabelEncoder()
encoded_color = le.fit_transform(data.color)
encoded_color = np.matrix.reshape(encoded_color,(371,1))
nor = Normalizer()
encoded_color = nor.fit_transform(encoded_color) 

data.drop(['color'],1,inplace=True)
dummy_data = np.concatenate((data,encoded_color),axis=1)

X = np.array(dummy_data)
y = copy.type
encoded_y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,encoded_y,test_size=0.4)

clf = RandomForestClassifier(n_estimators=300,n_jobs=5)

clf.fit(X_train,y_train)
pred = clf.predict(X_test)
accuracy = clf.score(X_test,y_test)
print(accuracy)
