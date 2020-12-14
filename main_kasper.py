# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:09:07 2020

@author: groes
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:23:18 2020

@author: groes
"""

# KASPER
import neural_network_kasper as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

### MAKING DATA
mnist = fetch_openml("mnist_784", version=1, cache=True)
X = mnist.data
y = mnist.target


### INITIALIZING MODEL
model = nn.Neural_network((100,100,50,10), 0.001)

X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = model.split_data(X, y, 0.2)

X_train_normalized = X_train / np.max(X_train)
X_test_normalized = X_test / np.max(X_test)

model.train(X_train_normalized, y_train_onehot, epochs = 5)


# Using trained model to make predictions
y_pred = model.predict(X_test_normalized)


# Calculating accuracy
correct_predictions = 0
index = 0
#y_test_int = int(y_test)
for row in y_pred:
    predicted = str(np.argmax(row))
    if predicted == y_test[index]:
        correct_predictions += 1
    index += 1

total_observations = len(y_test)

accuracy = (correct_predictions / total_observations) * 100
print("Accuracy is: {} %".format(accuracy))


