# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:15:48 2020

@author: groes
"""
import neural_network as nn
import numpy as np
import load_data
import utils

data = load_data.load_data()
X = data['data']
y = [float(ye) for ye in data['target']]

X_train, X_test, y_train, y_test = utils.split_data(X, y, 0.3)

unittest_mod = nn.new_neural_network(0.001)
unittest_mod.create_input_layer(784)
#unittest_mod.add_hidden_layer(216)
unittest_mod.add_hidden_layer(180)
unittest_mod.add_output_layer(10)
unittest_mod.new_train(X_train, y_train,3)

unittest_mod.accuracy_score(X_test, y_test)

y_test[0]



















###########################################
unittest_neurons_hidden_layers = np.array([40,40, 40]) #(16,16, 16, 16, 16, 16, 16, 16, 16, 16)
unittest_biases = np.ones(len(unittest_neurons_hidden_layers))
unittest_weight_range = (-1, 1)
unittest_lr = 0.001

#unittest_model = nn.Neural_network(
    unittest_neurons_hidden_layers, 
    unittest_biases, unittest_lr
    )

def unittest_sigmoid_function():
    x = np.array([[1,0,2,0,3,0,0,0,0,1],
                         [1,0,10,0,20,0,30,0,0,0],
                         [0,1,0,50,0,60,0,0,0,0]])
    
    result = unittest_model.sigmoid_activation_func(x)
    
    assert result.shape == (3, 10)
    assert result[0][1] == 0.5
    #good thing to assert would be sigmoid(0) = 1/2 ## this is a well known result
    #also 0<sigmoid(x)<1 for all x
    
def unittest_dsigmoid_function():
    x = np.array([[1,0,2,0,3,0,0,0,0,1],
                         [1,0,10,0,20,0,30,0,0,0],
                         [0,1,0,50,0,60,0,0,0,0]])
    
    result = unittest_model.dsigmoid(x)
    
    assert result.shape == (3, 10)
    assert result[0][1] == 0.25    


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
    
    # Experiementing - doesnt work
    #y_calc2 = 12
    #y2 = 5
    # Mocking dsigmoid
    #dsigmoid = Mock()
    #dsigmoid.return_value = 2
    #res = unittest_model.dcost_function(y_calc2, y2, g_input)
    
    result = unittest_model.dcost_function(y_calc, y, g_input)
    
    assert result.shape == g_input.shape == y.shape == y_calc.shape
    
    assert result[0][0] == 0.10499358540350662
    
    
    
    '''
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
    assert unittest_errors_twos == 37'''

unittest_cost_function()


def unittest_initialize_weights():
    #unittest_X_train = np.array([[21, 54, 76, 234, 25, 48, 28],
    #                        [6, 4, 2, 57, 423, 54, 43],
    #                        [43, 54, 76, 87, 35, 45, 23]])

    #unittest_Y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    unittest_model.split_data(data.data, data.target, 0.2)
    #unittest_mode.split_data()
    unittest_model.initialize_weight_matrices()
    
    number_of_input_features = unittest_model.X_train.shape[1] #unittest_X_train.shape[1]
    
    # Testing that there is a weight matrix for each connection between layers
    #assert len(unittest_model.list_of_weight_matrices) == unittest_number_of_hidden_layers + 1
    
    # Testing that the weight matrices have the correct number of weights
    assert unittest_model.list_of_weight_matrices[0].shape[1] == unittest_neurons_hidden_layers[0]
    assert unittest_model.list_of_weight_matrices[0].shape[0] == number_of_input_features
   
    # Testing that there are 10 connections between each neuron in the last hidden layer
    # and the output layer because there are 10 neurons in the output layer
    assert unittest_model.list_of_weight_matrices[-1].shape[1] == 10
    
    # Testing that there there are 16 sets of connections between last hidden layer
    # and output layer because there are 16 neurons in the last hidden layer
    assert unittest_model.list_of_weight_matrices[-1].shape[0] == unittest_neurons_hidden_layers[0]
    
    for matrix in unittest_model.list_of_weight_matrices:
        assert np.max(matrix) <= 1
        assert np.min(matrix) >= -1

    
unittest_initialize_weights()


def unittest_relu_activation_func():
    unnitest_X = [-2,4,-5,7,9]
    activated_result = unittest_model.relu_activation_func(unnitest_X)
    
    assert len(activated_result) == len(unnitest_X)
    assert list(activated_result) == [0,4,0,7,9]
    
 
unittest_relu_activation_func()

def unittest_calculate_activations():
    pass
    #feedforward basically just calls this a lot so don't need to assert anything
    #unittest_model.split_data(data.data, data.target, 0.2)
    #X_batch = unittest_model.X_train 
    #unittest_model.initialize_weight_matrices()
    
    
    #new_activations, product = unittest_model.calculate_activations(
    #    X_batch, unittest_model.list_of_weight_matrices[0])
    
    
    
unittest_calculate_activations()

def unittest_dcost_hidden_layer():
    #   def dcost_hidden_layer(self, err_layer_plus_one, weight_matric, g_input)
    
    err_layer_plus_one = np.array([np.ones(6),
                                  np.ones(6)*2])
    
    weight_matrix = np.array([np.ones(6),
                              np.ones(6)*2,
                              np.ones(6)*3,
                              np.ones(6)*4,
                              np.ones(6)*5])
    
    g_input = np.array([[0.12, -0.65, 0, -1, 1],
                             [0.03, -0.342, 0.54, -1, 1]])
    
    dcost = unittest_model.dcost_hidden_layer(err_layer_plus_one, weight_matrix, g_input) 
        
    assert dcost.shape == g_input.shape
    # For each rown in dcost, there's an element for each neuron in the hidden layer
    assert dcost.shape[1] == unittest_neurons_hidden_layers[0]
    
def unittest_weight_update():
         
    # def weight_update(self, err, activation):
    err = np.array([np.ones(6),
                    np.ones(6)*2])
    
    activations = [np.ones(unittest_neurons_hidden_layers[0]),
                   np.ones(unittest_neurons_hidden_layers[0])*2]
    
    result = unittest_model.weight_update(err, activations)
    
    assert len(result) == len(err) == len(activations)
    assert len(activations[0]) == len(result[0])
    assert len(err[0]) == len(result[0][0])

unittest_weight_update()
    
def unittest_feed_forward():
    unittest_model.split_data(data.data, data.target, 0.2)
    unittest_model.get_minibatch(40, unittest_model.X_train, unittest_model.y_train_onehot)
    unittest_model.initialize_weight_matrices()
    
    #activations_previous_layer = np.array([[10,6,4,2,64], [34,85,23,54,75]])
    
    #unittest_model.calculate_activations(
    #    weight_matrices[0], activations_previous_layer)
    
    
    unittest_model.feed_forward(unittest_model.X_batch)
    
    for i in range(len(unittest_model.activations)):
        assert np.max(unittest_model.activations[i]) <= 1
        assert np.min(unittest_model.activations[i]) >= -1
    
    assert unittest_model.activations[0].shape == (40, 3)
    assert unittest_model.activations[1].shape == (40, 3)
    assert unittest_model.activations[2].shape == (40, 10)
    assert len(unittest_model.activations) == unittest_number_of_hidden_layers +1
    assert len(unittest_model.g_inputs) == unittest_number_of_hidden_layers +1

unittest_feed_forward()      
        

def unittest_softmax_output():
    output = np.array([[1,3,5], [4,5,6]])
    softmax_output = unittest_model.hsoftmax(output)
    #softmax_output = np.round(softmax_output, 4)
    print(np.sum(softmax_output[1]))
    assert np.sum(softmax_output[0]) == 1.00000
    assert np.sum(softmax_output[1]) == 1.00000
    assert softmax_output[0][0] == round(0.01587624, 4)
    assert softmax_output[0][1] == round(0.11731043, 4)
    assert softmax_output[0][2] == round(0.86681333, 4)
    
#unittest_softmax_output()


def unittest_backprop():
    unittest_model.split_data(data.data, data.target, 0.2)
    unittest_model.initialize_weight_matrices()
    print(unittest_model.activations[-1])
    print(unittest_model.list_of_weight_matrices[-1][0])
    unittest_model.get_minibatch(40, unittest_model.X_train, unittest_model.y_train_onehot)
    #unittest_model.feed_forward(unittest_model.X_batch)    
    #unittest_model.backprop()
    #unittest_model.train(20)
    print(unittest_model.list_of_weight_matrices[-1][0])
    
    #for layer in range(len(unittest_model.list_of_weight_matrices)):
     #   assert len(unittest_model.delta_err[layer]) == len(unittest_model.list_of_weight_matrices[layer])

            
unittest_backprop()
#one epoch only changes the weights by last 4 decimal places

    
def unittest_one_hot():
    
    a = np.array([[0,1,2,3,2],[3,4,5,6,7,8]])
    b = unittest_model.one_hot_encoding(a)
    assert len(b[0]) == len(np.unique(a))
    assert len(b) == len(a)
    for i in b:
        assert sum(i) == 1
    assert b[0][0] == 1
    assert b[4][2] == 1

unittest_one_hot()


def unittest_predict():
    number_of_datapoints = 120
    
    # rows are datapoints
    X_test = unittest_model.X_test[:number_of_datapoints, :]
    #unittest_model.list_of_weight_matrices
    y_pred = unittest_model.predict(X_test)
    for i in y_pred:
        assert sum(i) > 0.99
        assert sum(i) < 1.11
    #would Ideally be 1 but good enough range for me
    
    
    
    # *10 because the last layer has 10 activations per datapoint
    assert y_pred.shape == (number_of_datapoints, 10)
    
    number_of_datapoints = 130
    # rows are datapoints
    X_test = unittest_model.X_test[:number_of_datapoints, :]
    #unittest_model.list_of_weight_matrices
    y_pred = unittest_model.predict(X_test)
    
    # *10 because the last layer has 10 activations per datapoint
    assert y_pred.shape == (number_of_datapoints, 10)
    
unittest_predict()  

    
def unittest_accuracy():
    y_pred = np.array([1,2,3,4,5,6,7,8,9])
    y_test = np.array([0,0,0,0,0,1,0,0,9])
    assert unittest_model.accuracy(y_pred, y_test) == 1

unittest_accuracy()

def unittest_evaluate():