# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:40:44 2020

@author: groes
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np

def split_data(X, y, test_size, normalise=True):
    """
    Splits and normalizes data
    
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    y_train_onehot = one_hot_encoding(y_train)
    y_test_onehot = one_hot_encoding(y_test)        
    
    if normalise:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        #X_train = np.apply_along_axis(normalise_input, 1, X_train)
        #X_test = np.apply_along_axis(normalise_input, 1, X_test)
    
    return X_train, X_test, y_train_onehot, y_test_onehot
    

def normalise_input(datapoint):
    Mu = sum(datapoint)/len(datapoint)
    SD = sum((datapoint-Mu)*(datapoint-Mu))/len(datapoint)
    znorm = (datapoint - Mu)/np.sqrt(SD + 0.0001)
    return znorm


def one_hot_encoding(data):
    onehot = [] 
    vector_length = len(np.unique(data))
    for i in data:
        vector = np.zeros(vector_length)
        vector[int(i)] = 1
        onehot.append(vector)
         
    return np.array(onehot)

def standardise(X):
        return X/np.max(X)