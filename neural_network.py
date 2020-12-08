# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:31 2020

@author: groes
"""
#I've made a change

import numpy as np
from sklearn.model_selection import train_test_split
import random
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
        # we can mutate the existing lists rather than appending them
        empty_list = []
        for i in range(number_of_hidden_layers + 1): # +1 to account for the output layer
            self.activations.append(empty_list)
        
        for i in range(number_of_hidden_layers + 1): # no +1 to account for the output layer
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
  
        self.y_test_onehot = self.one_hot_encoding(self.y_test)
        self.y_train_onehot = self.one_hot_encoding(self.y_train)        
        # Normalizing input feautres
        self.X_train = self.normalise_input(self.X_train)
        self.X_test = self.normalise_input(self.X_test)

        
        #return self.X_train, self.X_test, self.y_train, self.y_test, self.y_test_onehot

    def one_hot_encoding(self, data):
        
        onehot = [] 
        vector_length = len(np.unique(data))
        for i in data:
            vector = np.zeros(vector_length)
            vector[int(i)] = 1
            onehot.append(vector)
             
        return np.array(onehot)

    
    def normalise_input(self, data):
        normalised_data = data / np.max(data)
        return normalised_data
        
    def softmax(self, array):
        return np.exp(array)/np.sum(np.exp(array))
    
    def initialize_weight_matrices(self, initial_weight_range = (-1, 1)):
        
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
        number_of_input_nodes = self.X_train.shape[1]
        number_of_output_nodes = len(np.unique(self.y_train))
        
        list_of_weight_matrices = []
        
        
        # Creating weight matrix for connections from input layer to first hidden layer
        #weight_matrix_first_to_hidden = np.random.uniform(
        #    initial_weight_range[0], initial_weight_range[1],
        #    [self.number_of_nodes_per_hidden_layer[0], number_of_input_nodes]
        #    )
        weight_matrix_first_to_hidden = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [number_of_input_nodes, self.number_of_nodes_per_hidden_layer[0]]
            )
        

        list_of_weight_matrices.append(weight_matrix_first_to_hidden)
        
        # Creating weight matrices for the connections between hidden layers
        #for i in range(1, self.number_of_hidden_layers):
        #    matrix = np.random.uniform(
        #        initial_weight_range[0], initial_weight_range[1],
        #        [self.number_of_nodes_per_hidden_layer[i],
        #         self.number_of_nodes_per_hidden_layer[i]]
        #        )
        #    list_of_weight_matrices.append(matrix)
            
        for i in range(1, self.number_of_hidden_layers):
            matrix = np.random.uniform(
                initial_weight_range[0], initial_weight_range[1],
                [self.number_of_nodes_per_hidden_layer[i],
                 self.number_of_nodes_per_hidden_layer[i]]
                )
            list_of_weight_matrices.append(matrix)
        
        # Creating weight matrix for connections from last hidden layer to output layer
        #weight_matrix_hidden_to_output = np.random.uniform(
        #    initial_weight_range[0], initial_weight_range[1],
        #    [number_of_output_nodes, self.number_of_nodes_per_hidden_layer[-1]]
        #    )
        
        weight_matrix_hidden_to_output = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [self.number_of_nodes_per_hidden_layer[-1], number_of_output_nodes]
            )
        
        list_of_weight_matrices.append(weight_matrix_hidden_to_output)
        
        self.list_of_weight_matrices = list_of_weight_matrices

    def calculate_activations(self, activations_previous_layer, weight_matrix,
                               bias = 0, activation_func = "sigmoid"):
        
        product = np.matmul(activations_previous_layer, weight_matrix) + bias
        
        if activation_func == "sigmoid":
            new_activations = self.sigmoid_activation_func(product)
        else:
            new_activations = self.relu_activation_func(product)
        
        return new_activations, product
        
       
    
    def relu_activation_func(self, x):
        return np.maximum(0, x)
        
        
    def sigmoid_activation_func(self, x):
        #x = x.float()
        #x = np.exp(-x)
        #x = x + 1
        #x = 1/x
        sig = 1/(1 + np.exp(-x))
        return sig
    

    def dsigmoid(self, x):
        # self. keywords have been removed for the sake of experimenting with unit tests

        return self.sigmoid_activation_func(x)*(1-self.sigmoid_activation_func(x))
        

    def dcost_function(self, y_calc, y, g_input): 
        '''
        y_calc is the final activations. this is a 2D array (number of datapoints, number of output nodes), 
        every i is an array with the activations for each node calculated with the ith datapoint
        
        y is a 1D array of target values. length = number of nodes

        Parameters
        ----------
        y_hat : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        dsigmoid_result = self.dsigmoid(g_input)
        #for this to work data_point and y need to be np.array, NOT list or tuple etc
        return dsigmoid_result * (y_calc-y)
    # * gives elementwise multiplication, - gives elementwise subtraction
    
    def dcost_hidden_layer(self, err_layer_plus_one, weight_matrix, g_input):
        return self.dsigmoid(g_input)* np.matmul(err_layer_plus_one, weight_matrix.transpose()) 
    
    def weight_update(self, err, activation):
        weights = []
        for i in range(len(err)):
            weights.append(err[i] * activation[i][:,np.newaxis])
        return weights
            
    # not yet unit tested
    def get_minibatch(self, batch_size, X_train, y_train_onehot):
        # Hugh: I'm not sure np.ndarray is the right way to do this but I needed to use flatten
        batch_indicies = random.sample(range(len(self.X_train)), batch_size) # why have I 
        self.X_batch = X_train[batch_indicies]
        self.y_batch = y_train_onehot[batch_indicies]

        

    
        
    def feed_forward(self, X_batch):
        self.activations[0], self.g_inputs[0] = self.calculate_activations(
            X_batch, self.list_of_weight_matrices[0], bias = self.bias[0]  
            )
        
        for layer in range(1, len(self.activations)):     
            self.activations[layer], self.g_inputs[layer] = self.calculate_activations(
                self.activations[layer-1],
                self.list_of_weight_matrices[layer],
                bias = self.bias[layer-1]
                )
            
#for i in range(1, len(unittest_model.activations)): print(i)
        
#unittest_model.activations[2]
#len(unittest_model.g_inputs)


    def backprop(self):

            #input to this is already a batch    
        #H: when doing a batch g_inputs[-1] is a vactor, how does that square?
        self.delta_err[-1] =self.dcost_function(self.activations[-1], self.y_batch, self.g_inputs[-1])
        
        #delta_err[-1] is a 1D array the length of output nodes
        for layer in range(self.number_of_hidden_layers-1, 1, -1):

            #self.delta_err[layer] = dsigmoid(self.g_inputs[layer])* np.matmul(self.delta_err[layer+1], self.weight_matrices[layer].transpose()) 
            #the result of the matmul is a 2D array length (datapoints, g_inputs[layer])
            #think that both of these are the same size so its fine
            self.delta_err[layer] = self.dcost_hidden_layer(self.delta_err[layer+1], self.list_of_weight_matrices[layer], self.g_inputs[layer])
            
            #weight_update = []
            #for i in range(len(self.delta_err[layer])):
                
            #    weight_update.append(self.delta_err[layer][i] * self.activations[layer][i][:,np.newaxis])
            
            update = self.weight_update(self.delta_err[layer], self.activations[layer])
            
            self.list_of_weight_matrices[layer] += self.lr * sum(update)/len(self.y_batch)


            
    def train(self, epochs, batch_size = 40):
        self.batch_size = batch_size
        for epoch in range(epochs):
            self.get_minibatch(batch_size, self.X_train, self.y_train_onehot)
            self.feed_forward()
            self.backprop()
            
            
    def predict(self, X_test):
        # TO DO: 
            # run the preds through softmax
  
        # Here, we're not using get_minibatches because that method samples
        # random indices 
        y_pred = np.array([])
        from_index = 0
        to_index = self.batch_size
        iteration = 0
        while (len(y_pred) // 10) < len(X_test):
            iteration += 1
            # preventing indexing from going out of bounds
            if len(X_test)-(len(y_pred)//10) < self.batch_size:
                to_index = len(X_test)
            X_test_subset = X_test[from_index:to_index, :]

            self.feed_forward(X_test_subset)
            y_pred = np.append(y_pred, self.activations[-1])       
            from_index = to_index
            to_index = from_index + self.batch_size
            
        # Reshaping to make comparison with y_test easier
        y_pred_reshaped = np.reshape(y_pred, (len(y_pred)//10, 10))
        return y_pred_reshaped
    
    def evaluate(self, X_test):
        y_pred = self.predict(X_test)
        y_onehot = self.one_hot_encoding(self.y_test)
        why = y_pred - y_onehot
        why_squared = why*why
        MSE = (1/len(y_pred))*np.sum(why_squared)
        return MSE
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            



