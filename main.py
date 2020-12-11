# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:23:18 2020

@author: groes
"""
import load_data
import neural_network as nn

############ Calling the functions and class methods
mnist_dataset = load_data.load_data()

model = nn.Neural_network((100,100,50,10), (-0.1,-0.1,-0.1,0), 0.001)
model.split_data(mnist_dataset.data, mnist_dataset.target, test_size = 0.2)

model.train(3)

model.evaluate(model.X_test)
model.accuracy(model.predict(model.X_test), model.y_test_onehot)
