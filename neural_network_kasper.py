# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:08:36 2020

@author: groes
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:31 2020

@author: groes
"""
#make changes
# comment by kasper

import numpy as np
from sklearn.model_selection import train_test_split
import random
import sys

class Neural_network:
    # CHANGE: Removing bias from parameters because it should not be given as
    # argument, but be initialized as random matrix
    def __init__(self, number_of_nodes_per_hidden_layer, learning_rate):
        
        self.number_of_nodes_per_hidden_layer = number_of_nodes_per_hidden_layer # should be given as a tuple or list
        self.number_of_hidden_layers = len(number_of_nodes_per_hidden_layer)
                
        self.activations = [] # list that holds activivations for each hidden layer and the output layer
        self.g_inputs = []
        # CHANGE: Commenting out the next line because I'm gonna make bias a random matrix
        #self.bias = bias #
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
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = test_size)
            
            if len(np.unique(y_train)) == 10 and len(np.unique(y_test)) == 10:
                classes_10 = True
        
        #one hot encoding
  
        #self.y_test_onehot = self.one_hot_encoding(self.y_test)
        #self.y_train_onehot = self.one_hot_encoding(self.y_train)        
        # Normalizing input feautres
        #self.X_train = self.normalise_input(self.X_train)
        #self.X_test = self.normalise_input(self.X_test)
        
        
        y_test_onehot = self.one_hot_encoding(y_test)
        y_train_onehot = self.one_hot_encoding(y_train)        
        # Normalizing input feautres
        
        # CHANGE For now I'm leaving out the normalisation because it fucks things up
        #X_train = self.normalise_input(X_train)
        #X_test = self.normalise_input(X_test)
        
        return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot


    def one_hot_encoding(self, data):
        
        onehot = [] 
        vector_length = len(np.unique(data))
        for i in data:
            vector = np.zeros(vector_length)
            vector[int(i)] = 1
            onehot.append(vector)
             
        return np.array(onehot)

    

    def normalise_input(self, X_batch):
        Mu = sum(X_batch)/len(X_batch)
        s1 = X_batch - Mu
        s1 = s1*s1 + 0.0001
        SD = np.sqrt(s1)/len(X_batch)
        s2 = X_batch - Mu
        normalised_X_batch = s2/SD
        return normalised_X_batch

        
    def hsoftmax(self, array):
        new_array = []
        for datapoint in array:
            new_array.append(np.exp(datapoint)/np.sum(np.exp(datapoint)))
        new_array = np.array(new_array)
        return new_array

    
    # CHANGE Deleting the use of instance attributes - now using objects passed as arguments
    def initialize_weight_and_bias(self, initial_weight_range = (-1, 1), initial_bias_range = (-1, 1)):
        
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
        number_of_input_nodes = 784
        number_of_output_nodes = 10
        
        self.list_of_weight_matrices = []
        self.bias = []
        
        # Creating weight matrix for connections from input layer to first hidden layer
        #weight_matrix_first_to_hidden = np.random.uniform(
        #    initial_weight_range[0], initial_weight_range[1],
        #    [self.number_of_nodes_per_hidden_layer[0], number_of_input_nodes]
        #    )
        weight_matrix_first_to_hidden = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [number_of_input_nodes, self.number_of_nodes_per_hidden_layer[0]]
            )
        
        self.list_of_weight_matrices.append(weight_matrix_first_to_hidden)
        
        bias_matrix_first_to_hidden = np.random.uniform(
            initial_bias_range[0], initial_bias_range[1],
            [number_of_input_nodes, self.number_of_nodes_per_hidden_layer[0]]
            )
        
        self.bias.append(bias_matrix_first_to_hidden)

        
        
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
            self.list_of_weight_matrices.append(matrix)
            
        for i in range(1, self.number_of_hidden_layers):
            matrix = np.random.uniform(
                initial_bias_range[0], initial_bias_range[1],
                [self.number_of_nodes_per_hidden_layer[i-1],
                 self.number_of_nodes_per_hidden_layer[i]]
                )
            self.bias.append(matrix)
        
        # Creating weight matrix for connections from last hidden layer to output layer
        #weight_matrix_hidden_to_output = np.random.uniform(
        #    initial_weight_range[0], initial_weight_range[1],
        #    [number_of_output_nodes, self.number_of_nodes_per_hidden_layer[-1]]
        #    )
        
        weight_matrix_hidden_to_output = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [self.number_of_nodes_per_hidden_layer[-1], number_of_output_nodes]
            )
        
        self.list_of_weight_matrices.append(weight_matrix_hidden_to_output)

        bias_matrix_hidden_to_output = np.random.uniform(
            initial_weight_range[0], initial_weight_range[1],
            [self.number_of_nodes_per_hidden_layer[-1], number_of_output_nodes]
            )
        
        self.bias.append(bias_matrix_hidden_to_output)
        
        
    # CHANGE: Passing a bias matrix to this method now
    def calculate_activations(self, activations_previous_layer, weight_matrix,
                               bias, activation_func = "relu"):
        
        product = np.matmul(activations_previous_layer, weight_matrix)
        
        # CHANGE: Adding bias matrix here just for testing the effect on accuracy
        bias = np.random.uniform(-1, 1, product.shape)
        product = product + bias
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

        return np.array(dsigmoid_result * abs(y_calc-y))

    # * gives elementwise multiplication, - gives elementwise subtraction
    
    def dcost_hidden_layer(self, err_layer_plus_one, weight_matrix, g_input):
        return np.array(self.dsigmoid(g_input)* np.matmul(err_layer_plus_one, weight_matrix.transpose()))
    
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
        
        # We initialize activations and g_inputs here so that they are reset
        # every time feed_forward is called, which is once per batch
        activations = []
        g_inputs = []
        empty_list = []
        
        # There's one activations / g_input matrix per layer
        for i in range(self.number_of_hidden_layers + 1): # +1 to account for the output layer
            activations.append(empty_list)
        for i in range(self.number_of_hidden_layers + 1): # +1 to account for the output layer
            g_inputs.append(empty_list)
            
        # Calculating activations for first hidden layer    
        activations[0], g_inputs[0] = self.calculate_activations(
            X_batch, self.list_of_weight_matrices[0], bias = self.bias[0]  
            )
        
        # Calculating activations for the subsequent hidden layers
        for layer in range(1, len(activations)):     
            activations[layer], g_inputs[layer] = self.calculate_activations(
                activations[layer-1],
                self.list_of_weight_matrices[layer],
                bias = self.bias[layer-1])
        
        # Softmaxing the output layer
        activations[-1] = self.hsoftmax(g_inputs[-1])

        # QUESTION: Why are we returning g_inputs?         
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

    # CHANGE Im changing this so that datasets are passed as arguments and not accessed as instance attributes
    def train(self, X_train, y_train_onehot, epochs = 5, batch_size = 128): # QUESTION Should this method use y one hot? I think they should yes because the activations of the output layer is a vector of length 10 (1 for each digit)
        self.initialize_weight_and_bias()
        
        for epoch in range(epochs):
            print("Epoch: ")
            print(epoch+1)
            X_batches, y_batches = self.get_minibatch_new(batch_size, X_train, y_train_onehot)
            for batch in range(len(X_batches)):
                # CHANGE Im commenting out the next line because I think the datasets are all being normalized in split_data()
                #X_batch = self.normalise_input(X_batches[batch])
                batch_activations, batch_g_inputs  = self.feed_forward(X_batches[batch])
                self.backprop(X_batches[batch], y_batches[batch], batch_activations, batch_g_inputs)
                

            
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
    
    # CHANGE I'm changing this so that X_test and y_test are both passed as arguments instead of y_test_onehot being accessed as an instance attribute
    def evaluate(self, X_test, y_test_onehot): # QUESTION should y_test be one hot encoded?
        y_pred = self.predict(X_test)
        why = y_pred - y_test_onehot
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

            
######################################           

#a = np.array([0,1,1,1,0,0,0])
#np.argmax(a)


#a = np.array(np.zeros(400))
#a = a.reshape(40,10) 
#a.shape       
#b = np.array(np.zeros(160)) 
#b = b.reshape(16,10)     
#b.shape 
            
#np.matmul(a,b.transpose()).shape            

#a = 0.9
#b = round(a)            
#b
#abs(np.array([[1,2],[-3,4]]) - np.array([[1,1],[1,1]]))            
            
#for i in range(12,-1,-1):
 #   print(i)       
            

            
            
            



