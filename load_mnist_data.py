# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:55:26 2020

@author: groes
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def load_and_split_data(test_size):
    '''
    

    Parameters
    ----------
    test_size : float
        the proportion of the dataset you want to use as test data, e.g. 0.25

    Returns
    -------
    X_train : pandas series
        input features split into training data
    X_test : pandas series
        input features split into test data
    y_train : pandas series
        target features split into training data
    y_test : pandas series
        target features split into test data

    '''
    # The next line is borrowed from here:
    # https://github.com/ageron/handson-ml/issues/301
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    
    df = pd.DataFrame(data=np.c_[mnist['data'],mnist['target']],
                  columns=np.append(mnist['feature_names'],['target']))
    X = df.drop("target", axis=1)
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
        ) 
    
    return X_train, X_test, y_train, y_test





