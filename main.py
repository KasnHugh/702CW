# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:23:18 2020

@author: groes
"""
import load_data

import neural_network as nn

############ Calling the functions and class methods
mnist_dataset = load_data.load_data()

model = nn.Neural_network((40, 40, 16), (1,2,1), 0.1)

model.split_data(mnist_dataset.data, mnist_dataset.target, 0.3)

model.initialize_weight_matrices()

model.train(10)

model.evaluate(model.X_test)

model.lr
model.list_of_weight_matrices[0].shape
model.list_of_weight_matrices[1].shape
model.list_of_weight_matrices[2].shape
model.list_of_weight_matrices[3].shape

model.activations[-2].shape

len(model.delta_err[1])
