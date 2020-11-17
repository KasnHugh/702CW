# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:31 2020

@author: groes
"""

import numpy as np
from sklearn.model_selection import train_test_split
import sys

class Neural_network:
    def __init__(self, number_of_nodes_per_hidden_layer,
                 number_of_hidden_layers, bias):
        
        self.number_of_nodes_per_hidden_layer = number_of_nodes_per_hidden_layer # should be given as a tuple or list
        self.number_of_hidden_layers = number_of_hidden_layers
        
        if len(number_of_nodes_per_hidden_layer) != number_of_hidden_layers: 
            sys.exit("Number of hidden layers should match length of vector 'number of nodes per hidden layer' ")
            
        self.activations = [] # list that holds activivations for each hidden layer and the output layer
        self.bias = bias # should be given as tuple or list
        
        # Appending empty lists to the list 'activations' such that later,
        # I can mutate the existing lists rather than appending them
        empty_list = []
        for i in range(number_of_hidden_layers + 1): # +1 to account for the output layer
            self.activations.append(empty_list)


    def split_data(self, X, y, test_size):
        classes_10 = False
        
        # While loop ensures that the target variable splits contain all classes
        while classes_10 == False:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size = test_size
                )
            if len(np.unique(self.y_train)) == 10 and len(np.unique(self.y_test)) == 10:
                classes_10 = True
        
        
        # In feed_forward(), for each observation, the target variable needs to
        # be an array of length 10 where each index position represents a class
        # All values in the array is 0 except for the index position corresponding
        # to the class of y for the datapoint in question
        self.y_test_vectorized = []
        vector_length = len(np.unique(self.y_train))
        for i in self.y_train:
            vector = np.zeros(vector_length)
            vector[int(i)] = 1
            self.y_test_vectorized.append(vector)
             
        self.y_test_vectorized = np.array(self.y_test_vectorized)
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.y_test_vectorized
    
    def initialize_weight_matrices(self, X_train, y_train, initial_weight_range):
        
        '''
        
        Returns
        -------
        A list containing the followin ndarrays:
            
        weight_matrix_first_to_hidden : ndarray
            A matrix (ndarray) containing randomly initialized weights to be used for the connections between the input layer and the first hidden layer
        hidden layer weight matrices : ndarray
            Randomly initialzed weight matrix (ndarray) for each of the hidden layers in the network
        weight_matrix_hidden_to_output : ndarray
            A matrix (ndarray) containing randomly initialized weights to be used for the connections between the last hidden layer and the output layer

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
        
        # Using matmul instead of np.dot is recommended if multiplying a matrix by a matrix
        if activations_previous_layer.ndim == 2:
            product = np.matmul(weight_matrix, activations_previous_layer)
        else:
            product = np.dot(weight_matrix, activations_previous_layer) + bias
        
        if activation_func == "sigmoid":
            new_activations = self.__sigmoid_activation_func(product)
        else:
            new_activations = self.__relu_activation_func(product)
        
        return new_activations
        
    
    def __relu_activation_func(self, x):
        return np.maximum(0, x)
        
        
    def __sigmoid_activation_func(self, x):
        #x = x.float()
        x = np.exp(-x)
        x = x + 1
        x = 1/x
        return x

    def cost_function(self, y_hat, y):   
        return sum((y_hat-y)**2)
    
        
    def feed_forward(self, X_train, weight_matrices):
        '''
        TO DO:
            The method 
            should be called from the wrapper method once per batch - even if
            the number of batches is equal to the number of datapoints. 
            
            Does my cost_function return a vector?
            
            
        This function passes data forward in the network.
        This function should be called by some wrapper function. 

        Parameters
        ----------
        X_train : numpy array (1D or 2D)
            A numpy array with any amount of datapoints from the input feature training dataset
        y_test_vectorized : numpy array (1D)
            A numpy array representing the target variable 

        Returns
        -------
        errors : list
            A list containing n errors where n is the number of data points
            in the object passed as an argument to the method

        '''
        self.activations[0] = self.calculate_activiations(
            weight_matrices[0], X_train, bias = self.bias[0]  # Can I multiply the entire X_train matrix by a weight matrix with n weights where n is the number of data points in X_train=
            )
        #errors = []
        #for row in range(len(X_train)): # I think  I can make X_train a matrix instead of looping over it like this
            #self.activations[0] = self.calculate_activiations(weight_matrices[0], data_row)
        for layer in range(1, len(self.activations)):     
            self.activations[layer] = self.calculate_activiations(
                weight_matrices[layer],
                self.activations[layer-1],
                bias = self.bias[layer]
                )
            # if layer is the last layer (i.e. the output layer), calculate the error
            #if layer == len(self.activations):
             #   error = cost_function(self.activations[layer], y_train_vectorized, bias = self.bias[layer]) # I think the activations object needs to contain all the activations
                #errors.append(error) 
        #return self.activations
        
    # TO BE DEFINED
    def train(self, X_train, y_train, initial_weight_range = (-1, 1)):
        
        weight_matrices = self.initialize_weight_matrices(
            X_train, y_train, initial_weight_range)
    
        # TBD
        for row in X_train: 
            self.feed_forward(X_train, weight_matrices)
            error = self.cost_function(self.activations[-1])
            self.backprop(error)

















