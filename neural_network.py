# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:04:42 2020

@author: hugha
"""
from sklearn.model_selection import train_test_split
import numpy as np
import random
import utils
# TO DO:
# move preprocessing out of the class
# amend the train() method such that it call different activations functions based on layer parameter setting


class new_neural_network:
    def __init__(self, learning_rate):
        #self.hidden_layers = hidden_layers
        #self.hidden_plus_output = hidden_layers + [10]
        self.lr = learning_rate        
        #self.bias = [np.ones(layer) for layer in self.hidden_plus_output]
        
        
        self.layers = []

    # Adding methods for creating layers to make it easier to create layers 
    # with different parameters
    def create_input_layer(self, number_of_neurons, bias=1):
        # Storing parameters in dictionary inspired by suggestion in here to keep
        # logs of settings stored in json for record keeping
        # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn#:~:text=Too%20few%20neurons%20in%20a,%22memorize%22%20the%20training%20data
        parameters = {
            "layer_type" : "input",
            "bias" : bias,
            "number_of_neurons" : number_of_neurons,# making it parameterizable such that we can test the network on XOR
            "activations" : []
            }
        self.layers.append(parameters)
        
    # Consider changing default bias to None if it does not work with 0
    # but 0 is nicer for simplicity in other methods to avoid if statements
    # (i.e. we can just add bias without checking if layer has a bias)
    # Setting deafult activation func to relu as our experiments suggest Sigmiod  
    # is performing poorly for the MNIST classification problem
    def add_hidden_layer(self, number_of_neurons, activation_func="relu", bias=1):
        parameters = {
            "layer_type" : "hidden",
            "bias" : bias,
            "number_of_neurons" : number_of_neurons, 
            "activation_func" : activation_func,
            "weight_matrix" : self.initialize_weight_matrix(number_of_neurons, self.layers[-1]),
            "activations" : [],
            "g_inputs" : []
            }
        self.layers.append(parameters)

    def add_output_layer(self, number_of_neurons, activation_func="softmax"):
        parameters = {
            "layer_type" : "output",
            "number_of_neurons" : number_of_neurons, 
            "activation_func" : activation_func,
            "weight_matrix" : self.initialize_weight_matrix(number_of_neurons),
            "activations" : [],
            "g_inputs" : []
            }
        self.layers.append(parameters)
        
    
    def standardise(self, X):
        return X/np.max(X)
    
    # Moving normalise_input to utils.py
    #def normalise_input(self, datapoint):
        #Mu = sum(datapoint)/len(datapoint)
        #SD = sum((datapoint-Mu)*(datapoint-Mu))/len(datapoint)
        #znorm = (datapoint - Mu)/np.sqrt(SD + 0.0001)
        #return znorm

    def initialize_weight_matrix(self, number_of_neurons, initialization_method="gaussian"):
        '''
        np.matmul(activations_ prev_layer, weight matrix)

        weight[1][0][2] is the weight between node 0 in the first hidden layer 
            node 2 in the second hidden layer
            
        changed uniform to randn
        '''
        
        # If we have time, we can experiment with different ways of initializing
        if initialization_method == "xavier":
            pass
            # put return inside if statement so as to not run through the remaining 
            # if statements if condition is fulfilled 
        if initialization_method == "0":
            pass
            # put return inside if statement so as to not run through the remaining 
            # if statements if condition is fulfilled 
        #is this where bias goes?
        return np.random.randn(number_of_neurons, self.layers[-1]["number_of_neurons"])# + self.layers[-1]["bias"])
        
        
        
        #number_of_input_nodes = self.X_train.shape[1]
        #number_of_output_nodes = len(np.unique(self.y_train))
        #list_of_weight_matrices = []
        
        
        #Initialise first matrix
        #weight_matrix_first_to_hidden = np.random.randn(number_of_input_nodes, self.hidden_layers[0])
        #list_of_weight_matrices.append(weight_matrix_first_to_hidden)
        
        #Initialise all matricies between hidden layers
        #for i in range(1, len(self.hidden_layers)):
        ##    matrix = np.random.randn(self.hidden_layers[i-1],self.hidden_layers[i])
        #    list_of_weight_matrices.append(matrix)
        
        #initialise matrix that connects last hidden layer to output layer
        #weight_matrix_hidden_to_output = np.random.randn(self.hidden_layers[-1], number_of_output_nodes)
        #list_of_weight_matrices.append(weight_matrix_hidden_to_output)
        
        #self.list_of_weight_matrices = list_of_weight_matrices
    def new_train(self, X, y, epochs, batch_size = 128, optimiser = "Adam"):


        for epoch in epochs:
            
            mini_batches = self.get_minibatch(X,y, batch_size)
            
            for batch in mini_batches:
                X_batch = [x for (x,y) in batch]
                y_batch = [y for (x,y) in batch]
                
                
                #Forward Pass
                
                for layer in range(len(self.layers)):
                    
                    if self.layers[layer]["layer_type"] == "input":
                        self.layers[layer]["activations"] = X_batch
                    
                    else:
                        self.layers[layer]['activations'], self.layers[layer]['g_inputs'] = self.calculate_activations(self.layers[layer-1], self.layers[layer]['weight_matrix'])
                        
                
                #Backward Pass
                
                for layer in range((len(self.layers))-1, -1, -1):
                    
                    if self.layers[layer]["layer_type"] == "output":
                        #include this in the line below
                        self.layers[layer]['loss'] = self.get_output_loss(self.layers[layer], y_batch)
                        
                        self.layers[layer]['weight_update'] = self.weight_update(self.layers[layer]['loss'], self.layers[layer-1]['activations'])
                        
                        #This will take account of output activations eg softmax
                    
                    elif self.layers[layer]["layer_type"] == "input":
                        pass
                    
                    else:
                        self.layers[layer]['weight_update'] = self.backprop(self.layers[layer], self.layers[layer+1], optimiser)
                        
                for layer in self.layers:
                    if layer['layer_type'] == "hidden":
                        
                        layer['weight_matrix'] +=layer['weight_update']
                
                #Do I want to clear all unneede variables at this point?
                #for housekeeping reasons
                
        
                
            accuracy = self.accuracy_score(X)
            print("Epoch {}: accuracy = {}".format(epoch, accuracy))
        
    def get_minibatches(self, X, y, batch_size):
        training_data = [n for n in zip(X,y)]        
        random.shuffle(training_data)
        n = len(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
        return mini_batches
        
    def calculate_activations(self, prev_layer, current_layer):
        product = np.matmul(prev_layer['activations'], current_layer['weight_matrix']) + prev_layer['bias']
        
        if current_layer['activations_func'] == "sigmoid":
            new_activations = self.sigmoid_activation_func(product)
        elif current_layer['activations_func'] == "relu":
            new_activations = self.relu_activation_func(product)
        else:
            print("sigmoid or relu, you choose")
        
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
        
        
    def Adam(self, X_batch, y_batch, step_size = 0.001, rho1 = 0.9, rho2 = 0.999, stab = 10e-8):
        s = 0
        r = 0
        t = 0
        output_gradient = self.get_output_gradient(X_batch, y_batch)     
        t += 1
        
        s = rho1 * s + (1 - rho1) * output_gradient
        r = rho2 * r + (1 - rho2) * output_gradient * output_gradient #operator between grad is hadamar/elementwise operator

        s_hat = 

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
                self.SGD(x_batch, y_batch, batch_activations, batch_g_inputs)
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


    


    def hsoftmax(self, array):
        new_array = []
        for datapoint in array:
            new_array.append(np.exp(datapoint)/np.sum(np.exp(datapoint)))
        new_array = np.array(new_array)
        return new_array
    
    def SGD(self, X_batch, y_batch, activations, g_inputs):
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
# TO DO: Amend this such that it's calling util functions

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

