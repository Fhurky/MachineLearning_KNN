# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:34:35 2024

@author: furko
"""

import pandas as pd

data = pd.read_csv("KNNAlgorithmDataset.csv")

data['diagnosis'] = pd.factorize(data['diagnosis'])[0] + 1  # A, B to 1, 2
data = data.drop(['id'], axis = 1)


from sklearn.preprocessing import StandardScaler

first_column = data.iloc[:, 0]
other_columns = data.iloc[:, 1:]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(other_columns)

data_scaled = pd.DataFrame(data_scaled, columns=other_columns.columns)
data_scaled.insert(0, first_column.name, first_column)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data_scaled, train_size=0.33, random_state=53)

train_result = train.iloc[:, 0]
test_result = test.iloc[:, 0]

train = train.drop(['diagnosis'], axis = 1)
test = test.drop(['diagnosis'], axis = 1)

from sklearn.neighbors import KNeighborsClassifier

K = 5

KNN = KNeighborsClassifier(n_neighbors = K)
KNN.fit(train, train_result)

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(test_result, KNN.predict(test)))