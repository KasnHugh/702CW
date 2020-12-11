# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:23:18 2020

@author: groes
"""
import load_data

import neural_network as nn

############ Calling the functions and class methods
mnist_dataset = load_data.load_data()

model = nn.Neural_network((56,56, 56,10), (-0.1, 0.2 , 0, 0), 0.01)
model.split_data(mnist_dataset.data, mnist_dataset.target, 0.2)
model.initialize_weight_matrices()
model.train(20)
model.evaluate(model.X_test)

model.accuracy(model.predict(model.X_test), model.y_test_onehot)

model.feed_forward(model.X_batch)
model.activations[-1][1]
#network just isn't training
model.backprop()
model.get_
model.evaluate(model.X_test)

model.accuracy(model.predict(model.X_test), model.y_test)

len(model.y_test)

print(model.delta_err[-1])
