# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:31 2020

@author: groes
"""
#I've made a change

import numpy as np
from sklearn.model_selection import train_test_split
import sys

class Neural_network:
    def __init__(self, number_of_nodes_per_hidden_layer,
                 number_of_hidden_layers, bias, learning_rate):
        
        self.number_of_nodes_per_hidden_layer = number_of_nodes_per_hidden_layer # should be given as a tuple or list
        self.number_of_hidden_layers = number_of_hidden_layers
        
        if len(number_of_nodes_per_hidden_layer) != number_of_hidden_layers: #find a less dramatic function that exit, there are suggestions in moodle
            sys.exit("Number of hidden layers should match length of vector 'number of nodes per hidden layer' ")
            
        self.activations = [] # list that holds activivations for each hidden layer and the output layer
        self.g_inputs = []
        self.bias = bias # should be given as tuple or list
        self.delta_err = list(np.zeros(number_of_hidden_layers + 1))
        self.lr = learning_rate
        
        # Appending empty lists to the list 'activations' such that later,
        # I can mutate the existing lists rather than appending them
        empty_list = []
        for i in range(number_of_hidden_layers + 1): # +1 to account for the output layer
            self.activations.append(empty_list)
        
        for i in range(number_of_hidden_layers): # no +1 to account for the output layer
            self.g_inputs.append(empty_list)


    def split_data(self, X, y, test_size):
        classes_10 = False
        
        # While loop ensures that the target variable splits contain all classes
        while classes_10 == False:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size = test_size
                )
            if len(np.unique(self.y_train)) == 10 and len(np.unique(self.y_test)) == 10:
                classes_10 = True
        
        #one hot encoding
        self.y_test_onehot = [] 
        vector_length = len(np.unique(self.y_train))
        for i in self.y_train:
            vector = np.zeros(vector_length)
            vector[int(i)] = 1
            self.y_test_onehot.append(vector)
             
        self.y_test_onehot = np.array(self.y_test_onehot)
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.y_test_onehot
    
    def initialize_weight_matrices(self, X_train, y_train, initial_weight_range):
        
        '''
        
        Returns
        -------
        A list containing the followin ndarrays:
            
        weight_matrix_first_to_hidden : ndarray
            A matrix (ndarray) containing randomly initialized weights to be
            used for the connections between the input layer and the first hidden layer
        hidden layer weight matrices : ndarray
            Randomly initialzed weight matrix (ndarray) for each of the hidden
            layers in the network
        weight_matrix_hidden_to_output : ndarray
            A matrix (ndarray) containing randomly initialized weights to be 
            used for the connections between the last hidden layer and the output layer

        '''
        number_of_input_nodes = X_train.shape[1]
        number_of_output_nodes = len(np.unique(y_train))
        
        list_of_matrices = []
        
        # Creating weight matrix for connections from input layer to first hidden layer
        weight_matrix_first_to_hidden = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [self.number_of_nodes_per_hidden_layer[0], number_of_input_nodes]
            )
        list_of_matrices.append(weight_matrix_first_to_hidden)
        
        # Creating weight matrices for the connections between hidden layers
        for i in range(1, self.number_of_hidden_layers):
            matrix = np.random.uniform(
                initial_weight_range[0], initial_weight_range[1],
                [self.number_of_nodes_per_hidden_layer[i],
                 self.number_of_nodes_per_hidden_layer[i]]
                )
            list_of_matrices.append(matrix)
        
        # Creating weight matrix for connections from last hidden layer to output layer
        weight_matrix_hidden_to_output = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [number_of_output_nodes, self.number_of_nodes_per_hidden_layer[-1]]
            )
        
        list_of_matrices.append(weight_matrix_hidden_to_output)
        
        return list_of_matrices

    def calculate_activiations(self, weight_matrix, activations_previous_layer,
                               bias = 0, activation_func = "sigmoid"):
        
        product = np.dot(weight_matrix, activations_previous_layer) + bias
        
        if activation_func == "sigmoid":
            new_activations = self.__sigmoid_activation_func(product)
        else:
            new_activations = self.__relu_activation_func(product)
        
        return new_activations, product
        
       
    
    def relu_activation_func(self, x):
        return np.maximum(0, x)
        
        
    def sigmoid_activation_func(self, x):
        #x = x.float()
        x = np.exp(-x)
        x = x + 1
        x = 1/x
        return x
    
    def dsigmoid(self, x):
        z = self.__sigmoid_activation_func(x)*(1-self.__sigmoid_activation_func(x))
        return z

    def cost_function(self, y_hat, y):   
        return sum((y_hat-y)**2)
    
        
    def feed_forward(self, X_train, weight_matrices):

        self.activations[0], self.g_inputs[0] = self.calculate_activiations(
            weight_matrices[0], X_train, bias = self.bias[0]  
            )
        
        for layer in range(1, len(self.activations)):     
            self.activations[layer], self.g_inputs[layer] = self.calculate_activiations(
                weight_matrices[layer],
                self.activations[layer-1],
                bias = self.bias[layer]
                )
          

        
        #create an error matrix
    # TO BE DEFINED
    def train(self, X_train, y_train, initial_weight_range = (-1, 1)):
        
        weight_matrices = self.initialize_weight_matrices(
            X_train, y_train, initial_weight_range)
    
        # TBD
        for row in X_train: 
            self.feed_forward(X_train, weight_matrices)
            error = self.cost_function(self.activations[-1])
            self.backprop(error)

    def backprop(self, list_of_matrices):
        #this is a back prop for gradient descent, NOT SGD. need to change this
        self.delta_err.append(
            self.__dsigmoid(self.g_inputs[-1])*np.subtract(self.y_train, 
                                                           self.activations[-1]))
        
        
                
        for layer in range(self.number_of_hidden_layers-1, 1, -1):
            self.delta_err[layer] = np.dot(self.delta_err[layer + 1].transpose(),
                                           self.weight_matrices[layer])*self.__dsigmoid(self.g_inputs[layer])
            list_of_matrices[layer] += self.lr * np.dot(
                self.activations[layer],self.delta_err[layer + 1].transpose()) #H thinks list of matricies should be self







