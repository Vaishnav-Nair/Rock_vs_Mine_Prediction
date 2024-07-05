# -*- coding: utf-8 -*-
"""Rock vs Mine Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eEtBZ5Jv-iAHzIwTvIA0Sry2aLrPXEZ5

Importing Dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to pandas dataframe
sonar_data = pd.read_csv('/content/Dataset/Copy of sonar data.csv', header=None)

sonar_data.head()

#number of rows and columns
sonar_data.shape

#to get the statistical measures of the data
sonar_data.describe()

#to get how any mines and rocks are present in the data. data of rocks and mines are given in 60th column.
sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

#separtaing data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)

print(Y)



"""Training and Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)



"""Model Training using LOGISTIC REGRESSION"""

model = LogisticRegression()

#training the logistic regression model with training data
model.fit(X_train, Y_train)



"""Model Evaluation"""

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ',training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data: ',test_data_accuracy)



"""Making a Predictive system

"""

input_data = (0.0094,0.0333,0.0306,0.0376,0.1296,0.1795,0.1909,0.1692,0.1870,0.1725,0.2228,0.3106,0.4144,0.5157,0.5369,0.5107,0.6441,0.7326,0.8164,0.8856,0.9891,1.0000,0.8750,0.8631,0.9074,0.8674,0.7750,0.6600,0.5615,0.4016,0.2331,0.1164,0.1095,0.0431,0.0619,0.1956,0.2120,0.3242,0.4102,0.2939,0.1911,0.1702,0.1010,0.1512,0.1427,0.1097,0.1173,0.0972,0.0703,0.0281,0.0216,0.0153,0.0112,0.0241,0.0164,0.0055,0.0078,0.0055,0.0091,0.0067)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the input data array as we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)
print(prediction)

if (prediction[0]=='R'):
  print('This is a Rock')
else:
  print('This is a Mine')
