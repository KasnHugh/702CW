# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:15:48 2020

@author: groes
"""
import neural_network as nn
import numpy as np

unittest_number_of_hidden_layers = 10
unittest_neurons_hidden_layers = (16,16, 16, 16, 16, 16, 16, 16, 16, 16)
unittest_biases = (1,2,1)
unittest_weight_range = (-1, 1)
unittest_lr = 0.001

unittest_model = nn.Neural_network(
    unittest_neurons_hidden_layers, 
    unittest_number_of_hidden_layers, 
    unittest_biases, unittest_lr
    )

def unittest_cost_function():
    y_calc = np.array([[1,2,3,4,5,6,7,8,9,10],
                         [11,12,13,14,15,16,17,18,19,20],
                         [21,22,23,24,25,26,27,28,29,30]])
    
    y = np.array([[0,0,0,0,0,0,0,0,0,1],
                         [1,0,0,0,0,0,0,0,0,0],
                         [0,1,0,0,0,0,0,0,0,0]])
    
    g_input = np.array([[2,4,6,8,9,2,6,3,6,9],
                         [7,3,1,4,5,6,7,8,9,5],
                         [9,8,7,6,5,4,3,2,4,5]])
    
    result = unittest_model.dcost_function(y_calc, y, g_input)
    
    assert result.shape == (3, 10)
    
    assert result[0][0] == 0.10499358540350662
    
    
    
    
    #test_activations = np.zeros(10)
    unittest_activations_zeros = np.zeros(10)
    unittest_activations_ones = np.ones(10)
    unittest_activations_twos = np.ones(10)+1
    unittest_y_train = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    unittest_errors_zeros = unittest_model.dcost_function(
        unittest_y_train, unittest_activations_zeros)
    unittest_errors_ones = unittest_model.dcost_function(
        unittest_activations_ones, unittest_y_train
        )
    unittest_errors_twos = unittest_model.dcost_function(
        unittest_activations_twos, unittest_y_train
        )
    
    assert unittest_errors_zeros == 1
    assert unittest_errors_ones == 9
    assert unittest_errors_twos == 37

unittest_cost_function()

def unittest_initialize_weights():
    unittest_X_train = np.array([[21, 54, 76, 234, 25, 48, 28],
                            [6, 4, 2, 57, 423, 54, 43],
                            [43, 54, 76, 87, 35, 45, 23]])

    unittest_Y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    weights = unittest_model.initialize_weight_matrices(unittest_X_train,
                                                        unittest_Y_train,
                                                        unittest_weight_range)
    number_of_input_features = unittest_X_train.shape[1]
    
    # Testing that there is a weight matrix for each connection between layers
    assert len(weights) == unittest_number_of_hidden_layers + 1
    
    # Testing that the weight matrices have the correct number of weights
    assert weights[0].shape[0] == unittest_neurons_hidden_layers[0]
    assert weights[0].shape[1] == number_of_input_features
   
    # Testing that there are 10 connections between each neuron in the last hidden layer
    # and the output layer because there are 10 neurons in the output layer
    assert weights[-1].shape[0] == 10
    
    # Testing that there there are 16 sets of connections between last hidden layer
    # and output layer because there are 16 neurons in the last hidden layer
    assert weights[-1].shape[1] == unittest_neurons_hidden_layers[0]
    
    for matrix in weights:
        assert np.max(matrix) <= 1
        assert np.min(matrix) >= -1

    
unittest_initialize_weights()


def unittest_relu_activation_func():
    unnitest_X = [-2,4,-5,7,9]
    activated_result = unittest_model.relu_activation_func(unnitest_X)
    
    assert len(activated_result) == len(unnitest_X)
    assert list(activated_result) == [0,4,0,7,9]
    
 
unittest_relu_activation_func()