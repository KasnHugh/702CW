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
    def __init__(self, number_of_nodes_per_hidden_layer, bias, learning_rate):
        
        self.number_of_nodes_per_hidden_layer = number_of_nodes_per_hidden_layer # should be given as a tuple or list
        self.number_of_hidden_layers = len(number_of_nodes_per_hidden_layer)
                
        self.activations = [] # list that holds activivations for each hidden layer and the output layer
        self.g_inputs = []
        self.bias = bias # should be given as tuple or list
        self.delta_err = []
        self.lr = learning_rate
        
        # Appending empty lists to the list 'activations' such that later,
        # we can mutate the existing lists rather than appending them
        empty_list = []
        for i in range(self.number_of_hidden_layers + 1): # +1 to account for the output layer
            self.activations.append(empty_list)
        
        for i in range(self.number_of_hidden_layers + 1): # no +1 to account for the output layer
            self.g_inputs.append(empty_list)
            
        for i in range(self.number_of_hidden_layers + 1): # no +1 to account for the output layer
            self.delta_err.append(empty_list)



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
        #self.X_train = self.normalise_input(self.X_train)
        self.X_test = self.normalise_input(self.X_test)


    def one_hot_encoding(self, data):
        
        onehot = [] 
        vector_length = len(np.unique(data))
        for i in data:
            vector = np.zeros(vector_length)
            vector[int(i)] = 1
            onehot.append(vector)
             
        return np.array(onehot)

    
    #def normalise_input(self, data):
     #   normalised_data = data / np.max(data)
      #  return normalised_data
    
    def normalise_input(self, X):
        Mu = sum(X)/len(X)
        Xu = X-Mu
        SD = sum(Xu*Xu)
        SD = np.sqrt(SD + 0.01)
        return (X-Mu)/SD
        
        
        
    def hsoftmax(self, array):
        new_array = []
        for datapoint in array:
            new_array.append(np.exp(datapoint)/np.sum(np.exp(datapoint)))
        new_array = np.array(new_array)
        return new_array
    
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
                [self.number_of_nodes_per_hidden_layer[i-1],
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
        '''
        dsigmoid_result = self.dsigmoid(g_input)
        #for this to work data_point and y need to be np.array, NOT list or tuple etc
        return dsigmoid_result * abs(y_calc-y)
    # * gives elementwise multiplication, - gives elementwise subtraction
    
    def dcost_hidden_layer(self, err_layer_plus_one, weight_matrix, g_input):
        return self.dsigmoid(g_input)* np.matmul(err_layer_plus_one, weight_matrix.transpose()) 
    
    def weight_update(self, err, activation):
        weights = []
        for i in range(len(err)):
            weights.append(err[i] * activation[i][:,np.newaxis])
        return weights
            
    # not yet unit tested
   # def get_minibatch(self, batch_size, X_train, y_train_onehot):
    #    batch_indicies = random.sample(range(len(self.X_train)), batch_size)
     #   self.X_batch = X_train[batch_indicies]
      #  self.y_batch = y_train_onehot[batch_indicies]

        
    def get_minibatch_new(self, batch_size, X, y_onehot):
        full_index = range(len(X))
        full_index = [i for i in full_index]
        random.shuffle(full_index)            
        list_of_batch_indicies = []
        
        spares = len(X) % batch_size
        spares_batch_indicies = []
        for x in range(spares):
            spares_batch_indicies.append(full_index.pop())
        if len(spares_batch_indicies)>0:
            list_of_batch_indicies.append(spares_batch_indicies)
        
        number_of_full_size_minibatches = (len(X)-spares)//batch_size
        for minibatch in range(number_of_full_size_minibatches):
            batch_indicies = []
            for point in range(batch_size):
                batch_indicies.append(full_index.pop())
            list_of_batch_indicies.append(batch_indicies)
        
        
        X_batches = []
        y_batches = []
        for batch in list_of_batch_indicies:
            X_batch = X[batch]
            X_batches.append(X_batch)
            y_batch = y_onehot[batch]
            y_batches.append(y_batch)

        return X_batches, y_batches

        
    def feed_forward(self, X_batch):
        activations = []
        g_inputs = []
        empty_list = []
        for i in range(self.number_of_hidden_layers + 1): # +1 to account for the output layer
            activations.append(empty_list)
        
        for i in range(self.number_of_hidden_layers + 1): # no +1 to account for the output layer
            g_inputs.append(empty_list)
        #activations and g_inputs shouldn't be attributes
        activations[0], g_inputs[0] = self.calculate_activations(
            X_batch, self.list_of_weight_matrices[0], bias = self.bias[0]  
            )
        
        for layer in range(1, len(activations)):     
            activations[layer], g_inputs[layer] = self.calculate_activations(
                activations[layer-1],
                self.list_of_weight_matrices[layer],
                bias = self.bias[layer-1]
                )
        activations[-1] = self.hsoftmax(g_inputs[-1])        
        return activations, g_inputs


    def backprop(self, X_batch, y_batch, activations, g_inputs):
        empty_list = []
        delta_err = []
        for i in range(self.number_of_hidden_layers + 1): # no +1 to account for the output layer
            delta_err.append(empty_list)
        delta_err[-1] = self.dcost_function(activations[-1], y_batch, g_inputs[-1])
        
        for layer in range(len(delta_err)-2,-1, -1):
            delta_err[layer] = self.dcost_hidden_layer(delta_err[layer+1], self.list_of_weight_matrices[layer+1], g_inputs[layer])
            
        for layer in range(len(delta_err)-1,0, -1):           
            update = self.weight_update(delta_err[layer], activations[layer-1])
            self.list_of_weight_matrices[layer] -= self.lr * sum(update) / len(y_batch)
                
        self.list_of_weight_matrices[0] += self.lr * sum(self.weight_update(delta_err[0], X_batch))/len(y_batch)     


    def train(self, epochs, batch_size = 128):
        self.initialize_weight_matrices()
        for epoch in range(epochs):
            X_batches, y_batches = self.get_minibatch_new(batch_size, self.X_train, self.y_train_onehot)
            for batch in range(len(X_batches)):
                X_batch = self.normalise_input(X_batches[batch])
                batch_activations, batch_g_inputs  = self.feed_forward(X_batch)
                self.backprop(X_batch, y_batches[batch], batch_activations, batch_g_inputs)
                
                
            
            
    def predict(self, X_test):
        train_activations = []
        empty_list = []
        for i in range(self.number_of_hidden_layers + 1): # +1 to account for the output layer
            train_activations.append(empty_list)
            
        g_input_activations = []
        empty_list = []
        for i in range(self.number_of_hidden_layers + 1): # +1 to account for the output layer
            g_input_activations.append(empty_list)
            
        
        train_activations[0], _ = self.calculate_activations(
            X_test, self.list_of_weight_matrices[0], bias = self.bias[0]  
            )
        
        for layer in range(1, len(train_activations)):     
            train_activations[layer], g_input_activations[layer] = self.calculate_activations(
                train_activations[layer-1],
                self.list_of_weight_matrices[layer],
                bias = self.bias[layer-1]
                )
            
        outputs = self.hsoftmax(g_input_activations[-1])
        return outputs
    
    def evaluate(self, X_test):
        y_pred = self.predict(X_test)
        why = y_pred - self.y_test_onehot
        why_squared = why*why
        MSE = (1/len(y_pred))*np.sum(why_squared)
        return MSE
    
    def accuracy(self, y_pred, y):
        accuracy_sum = 0
        for datapoint in range(len(y)):
            if np.argmax(y[datapoint]) == np.argmax(y_pred[datapoint]):
                accuracy_sum += 1
        
        accuracy = accuracy_sum/len(y)
        return accuracy
            
            
            

            
            
            
            
            
            
            
            
            



