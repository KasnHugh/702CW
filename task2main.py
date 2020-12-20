# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:23:18 2020

@author: groes
"""
import neural_network as nn
import numpy as np
import load_data
import utils

data = load_data.load_data()
X = data['data']
y = data['target']

X_train, X_test, X_val, y_train, y_test, y_val = utils.split_data(X, y, 0.2)


unittest_mod = nn.new_neural_network(0.001)
unittest_mod.create_input_layer(784)
unittest_mod.add_hidden_layer(128)
unittest_mod.add_hidden_layer(256)
unittest_mod.add_hidden_layer(128)
unittest_mod.add_output_layer(10)
unittest_mod.new_train(X_train, X_val, y_train, y_val, 50 ,batch_size = 32, optimiser= "SGD", stopping_criterion= "xent")


unittest_mod.accuracy_score(X_test, y_test, cm = True)
