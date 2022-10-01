# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 18:53:48 2022

@author: ENGINEER
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pickle

df = pd.read_csv(r"F:\ML\new_dataset.csv")
df.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['classes'])
df['classes']=le.transform(df['classes'])
le.fit(df['classes'])
df['classes']=le.transform(df['classes'])
# print(df)

df.info()
df.shape

# = df.iloc[:, [0,1,2,3,4,5,6,7,8]]
#X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
X = df[['green_abs_deviation', 'red_abs_deviation']]
Y = df.iloc[:,15]

Y.head(130)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

print("Training data : ",X_Train.shape)
print("Testing data : ",X_Test.shape)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

from sklearn.svm import SVC
classifier = SVC(C=5,kernel = 'rbf')
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_Test, Y_Pred))


Y_Pred

from sklearn import metrics
print('Accuracy Score: ')

print(metrics.accuracy_score(Y_Test,Y_Pred))


pickle.dump(classifier, open('model.pkl', 'wb'))


































