# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:04:42 2020

@author: hugha
"""
from sklearn.model_selection import train_test_split
import numpy as np
import random

#move preprocessing out of the class


class new_neural_network:
    def __init__(self, hidden_layers, learning_rate):
        self.hidden_layers = hidden_layers
        self.hidden_plus_output = hidden_layers + [10]
        self.lr = learning_rate        
        self.bias = [np.ones(layer) for layer in self.hidden_plus_output]

    

    #split preprocessing from NN
    def split_data(self, X, y, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size = test_size)
        
        self.y_train_onehot = self.one_hot_encoding(self.y_train)
        self.y_test_onehot = self.one_hot_encoding(self.y_test)        
        #input shape is 784
        #output shape is 10
        self.X_train = self.X_train/255
        self.X_test = self.X_test/255   

    def one_hot_encoding(self, data):
        
        onehot = [] 
        vector_length = len(np.unique(data))
        for i in data:
            vector = np.zeros(vector_length)
            vector[int(i)] = 1
            onehot.append(vector)
             
        return np.array(onehot)     
    
    def standardise(self, X):
        return X/np.max(X)
    
    def normalise_input(self, datapoint):
        Mu = sum(datapoint)/len(datapoint)
        SD = sum((datapoint-Mu)*(datapoint-Mu))/len(datapoint)
        znorm = (datapoint - Mu)/np.sqrt(SD + 0.0001)
        return znorm

    def initialize_weight_matrices(self, initial_weight_range = (-1, 1)):
        '''
        np.matmul(activations_ prev_layer, weight matrix)

        weight[1][0][2] is the weight between node 0 in the first hidden layer 
            node 2 in the second hidden layer
            
        changed uniform to randn
        '''
        number_of_input_nodes = self.X_train.shape[1]
        number_of_output_nodes = len(np.unique(self.y_train))
        list_of_weight_matrices = []
        
        
        #Initialise first matrix
        weight_matrix_first_to_hidden = np.random.randn(number_of_input_nodes, self.hidden_layers[0])
        list_of_weight_matrices.append(weight_matrix_first_to_hidden)
        
        #Initialise all matricies between hidden layers
        for i in range(1, len(self.hidden_layers)):
            matrix = np.random.randn(self.hidden_layers[i-1],self.hidden_layers[i])
            list_of_weight_matrices.append(matrix)
        
        #initialise matrix that connects last hidden layer to output layer
        weight_matrix_hidden_to_output = np.random.randn(self.hidden_layers[-1], number_of_output_nodes)
        list_of_weight_matrices.append(weight_matrix_hidden_to_output)
        
        self.list_of_weight_matrices = list_of_weight_matrices

        
    def train(self, X, y, epochs, batch_size = 128):
        self.initialize_weight_matrices()
        training_data = [n for n in zip(X,y)]
        random.shuffle(training_data)
        for epoch in range(epochs):
            n = len(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in mini_batches:
                x_batch = [x for (x,y) in batch]
                y_batch = [y for (x,y) in batch]
                batch_activations, batch_g_inputs  = self.feed_forward(x_batch)
                self.backprop(x_batch, y_batch, batch_activations, batch_g_inputs)
            print("Epoch {}".format(epoch))
        
    def feed_forward(self, X_batch):
        activations = []
        g_inputs = []
        empty_list = []
        
        #this means activations keeps all layers, input to output
        activations.append(X_batch)
        #this makes sure g_inputs[layer] and activations[layer] is the same layer, wont need to call g_inputs[0]
        g_inputs.append(np.zeros(np.array(X_batch).shape))
                              
        for i in range(len(self.hidden_layers) + 1): # +1 to account for the output layer
            activations.append(empty_list)
        
        for i in range(len(self.hidden_layers) + 1): # no +1 to account for the output layer
            g_inputs.append(empty_list)
            
        for layer in range(1, len(activations)):     
            activations[layer], g_inputs[layer] = self.calculate_activations(
                activations[layer-1],
                self.list_of_weight_matrices[layer-1],
                bias = self.bias[layer-1])
        
        #softmax output layer
        activations[-1] = self.hsoftmax(g_inputs[-1])        
        return activations, g_inputs

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

    def hsoftmax(self, array):
        new_array = []
        for datapoint in array:
            new_array.append(np.exp(datapoint)/np.sum(np.exp(datapoint)))
        new_array = np.array(new_array)
        return new_array
    
    def backprop(self, X_batch, y_batch, activations, g_inputs):
        empty_list = []
        delta_err = []
        for i in range(len(self.hidden_layers) + 2): # no +2 to account for the output layer and input layer
            delta_err.append(empty_list)
        
        #part of derivative of error at ouput layer
        delta_err[-1] = self.cost_prime(activations[-1], y_batch, g_inputs[-1])
        
        #looping backwards, starting at second last layer,backpropogating the error
        for layer in range(len(delta_err)-2,-1, -1):
            delta_err[layer] = self.cost_prime_hidden(delta_err[layer+1], self.list_of_weight_matrices[layer], g_inputs[layer])
        
        #creating the update for the weights at each layer
        for layer in range(len(delta_err)-1,0, -1):           
            update = self.weight_update(delta_err[layer], activations[layer-1])
            self.list_of_weight_matrices[layer-1] -= self.lr * sum(update) / len(y_batch)
         
        for layer in range(len(self.bias)-1,1, -1):         
            bias_update = sum(delta_err[layer+1])/len(y_batch)
            self.bias[layer] -= self.lr *bias_update
        
        
    #def relu_prime(self, g_input):
     #   new_g_inputs = [0 if x <= 0 else 1 for x in a]
      #  return new_g_inputs
            
    def cost_prime(self, y_calc, y, g_input): 
        dsigmoid_result = self.sigmoid_prime(g_input)
        #for this to work data_point and y need to be np.array, NOT list or tuple etc

        return np.array(dsigmoid_result * abs(y_calc-y))

    def cost_prime_hidden(self, err_layer_plus_one, weight_matrix, g_input):
        return np.array(self.sigmoid_prime(g_input)* np.matmul(err_layer_plus_one, weight_matrix.transpose()))
    
    def sigmoid_prime(self, x):
        return self.sigmoid_activation_func(x)*(1-self.sigmoid_activation_func(x))   

    def weight_update(self, err, activation):
        weights = []
        for i in range(len(err)):
            weights.append(err[i] * activation[i][:,np.newaxis])
        return weights



    
    def evaluate(self, X_test, y_test_onehot):
        y_pred, _ = self.feed_forward(X_test)
        y_pred = y_pred[-1]
        why = y_pred - y_test_onehot
        why_squared = why*why
        MSE = (1/len(y_pred))*np.sum(why_squared)
        return MSE
    

    def accuracy(self, X_test, y):
        y_pred, _ = self.feed_forward(X_test)
        y_pred = y_pred[-1]
        accuracy_sum = 0
        for datapoint in range(len(y)):
            if np.argmax(y[datapoint]) == np.argmax(y_pred[datapoint]):
                accuracy_sum += 1
        
        accuracy = accuracy_sum/len(y)
        return accuracy


#####################################

from sklearn.datasets import fetch_openml


### GETTING DATA
#mnist = fetch_openml("mnist_784", version=1, cache=True)
#X = mnist.data
#y = mnist.target

model = new_neural_network([100,100,100], 0.001)
model.split_data(X, y, test_size = 0.2)
#print(model.bias)
#print(model.y_train_onehot[0][0])
#y[0].dtype
model.train(model.X_train, model.y_train_onehot, 3)

model.evaluate(model.X_test, model.y_test_onehot)
model.accuracy(model.X_test, model.y_test_onehot)
#the first g_input and first activation is all zeros

#np.unique(model.y_test)

model.X_train.shape[1]
y_pred,_ = model.feed_forward(model.X_test)
y_pred = y_pred[-1]
# Calculating accuracy
correct_predictions = 0
index = 0
#y_test_int = int(y_test)
for row in y_pred:
    predicted = str(np.argmax(row))
    if predicted == model.y_test[index]:
        correct_predictions += 1
    index += 1

total_observations = len(model.y_test)

accuracy = (correct_predictions / total_observations) * 100
print("Accuracy is: {} %".format(accuracy))
    
    
##################################


#a = [[1,1],[1,3]]
np.unique(model.y_train[0:9])
def one_hot_new(data):
    one_hot_labels = []
    for item in data:
        start = np.zeros(10)
        start[int(item)] += 1
        one_hot_labesl.append(start)
    return one_hot_labels
#err = np.zeros((128,10))
#activation = np.zeros((128,30))

#nn = new_neural_network((2,3,4))
#update = np.array(nn.weight_update(err, activation))
#update.shape
#update = sum(update)
#update.shape





#use this list comprehension when calling X_batch
#def normalise_input(datapoint):
#    Mu = sum(datapoint)/len(datapoint)
#    SD = sum((datapoint-Mu)*(datapoint-Mu))/len(datapoint)
#    znorm = (datapoint - Mu)/np.sqrt(SD + 0.0001)
#    return znorm


#Xbatch = np.array([[0,1,2,3,4,5],[1,0,1,0,1,1]])
#Xbatch = [normalise_input(x) for x in Xbatch]
#Xbatch

#for layer in range(5, -1, -1):
#    print(layer)
    
    
#sizes = [784,40,40,40,10]
#new sizes = [(x,y) for x, y in zip(sizes[-1:], sizes[1:)]
##    print(x,y)s[-1:]
#batch_size = 3
#X = np.array([[0,1,2,3,4,5,6,7,8,9],
 #             [0,1,2,3,4,5,6,7,8,9],
  #            [0,1,2,3,4,5,6,7,8,9],
   #           [0,1,2,3,4,5,6,7,8,9],
    #          [0,1,2,3,4,5,6,7,8,9],
     #         [0,1,2,3,4,5,6,7,8,9],
      #        [0,1,2,3,4,5,6,7,8,9],
       #       [0,1,2,3,4,5,6,7,8,9],
        #      [0,1,2,3,4,5,6,7,8,9],
         #     [0,1,2,3,4,5,6,7,8,9]])
#n = len(X)

#mini_batches = [X[k:k+batch_size] for k in range(0, n, batch_size)]
#mini_batches

#a = np.zeros((4,3))
#b = np.ones(4)
#list = [x for x in (zip(a,b))]
#list
#list = [np.zeros(x) for x in range(4)]
#list + [10]
#print(list)
#a = [[-1,-1],[0,1],[-2,2],[3,4]]
#def signe(a):
 #   if a>0:
  #      1
   # else:
    #    0
#signe(np.array([-1,0,1]))

