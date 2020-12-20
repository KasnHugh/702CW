# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:04:42 2020

@author: hugha
"""

import numpy as np
import random


class new_neural_network:
    def __init__(self, learning_rate):
        #self.hidden_layers = hidden_layers
        #self.hidden_plus_output = hidden_layers + [10]
        self.lr = learning_rate        
        #self.bias = [np.ones(layer) for layer in self.hidden_plus_output]
        
        
        self.layers = []

    # Adding methods for creating layers to make it easier to create layers 
    # with different parameters
    def create_input_layer(self, number_of_neurons, bias=0.0001):
        # Storing parameters in dictionary inspired by suggestion in here to keep
        # logs of settings stored in json for record keeping
        # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn#:~:text=Too%20few%20neurons%20in%20a,%22memorize%22%20the%20training%20data
        parameters = {
            "layer_type" : "input",
            "bias" : bias,
            "number_of_neurons" : number_of_neurons,# making it parameterizable such that we can test the network on XOR
            "activations" : [],
            "bias_update" : []
            }
        self.layers.append(parameters)
        

    # Setting deafult activation func to relu as our experiments suggest Sigmiod  
    # is performing poorly for the MNIST classification problem
    def add_hidden_layer(self, number_of_neurons, activation_func="relu", bias=0.0001):
        parameters = {
            "layer_type" : "hidden",
            "bias" : bias,
            "number_of_neurons" : number_of_neurons, 
            "activation_func" : activation_func,
            "weight_matrix" : self.initialize_weight_matrix(number_of_neurons),
            "activations" : [],
            "g_inputs" : [],
            "s" : 0,
            "r" : 0,
            "delta" : [],
            "weight_update" : [],
            "bias_update" : []
            }
        self.layers.append(parameters)

    def add_output_layer(self, number_of_neurons, activation_func="softmax"):
        parameters = {
            "layer_type" : "output",
            "number_of_neurons" : number_of_neurons, 
            "activation_func" : activation_func,
            "weight_matrix" : self.initialize_weight_matrix(number_of_neurons),
            "activations" : [],
            "g_inputs" : [],
            "s" : 0,
            "r" : 0,
            "delta" : [],
            "weight_update" : []
            }
        self.layers.append(parameters)

    def initialize_weight_matrix(self, number_of_neurons, initialization_method="gaussian"):

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
        return np.random.uniform(-1,1,(self.layers[-1]["number_of_neurons"],number_of_neurons))
        
        
    def new_train(self, X, y, epochs, batch_size = 128, optimiser = "Adam"):
        print("Training Started")
        for epoch in range(epochs):
            
            mini_batches = self.get_minibatches(X,y, batch_size)           
            for batch in mini_batches:
                X_batch = [x for (x,y) in batch]
                y_batch = [y for (x,y) in batch]
                #print("forward pass started")
                self.forward_pass(X_batch)
                #print("backward pass started")
                self.backward_pass(y_batch, optimiser)
            
            self.forward_pass(X)                
            accuracy = self.accuracy_score(X, y)
            xent = self.xent(self.layers[-1]['activations'], y)
            print("Epoch {}: loss {}, accuracy = {}".format(epoch,xent, accuracy))

        
    def get_minibatches(self, X, y, batch_size):
        training_data = [n for n in zip(X,y)]        
        random.shuffle(training_data)
        n = len(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
        return mini_batches
        
    
    def forward_pass(self, X):
        for layer in range(len(self.layers)):
            if self.layers[layer]["layer_type"] == "input":
                self.layers[layer]["activations"] = X                    
            else:
                self.layers[layer]['activations'], self.layers[layer]['g_inputs'] = self.calculate_activations(self.layers[layer-1], self.layers[layer])
         
    def backward_pass(self, y_batch, optimiser):
        #calculating the weight and bias updates
        for layer in range((len(self.layers))-1, -1, -1):
                    
            #output
            if self.layers[layer]["layer_type"] == "output":
                #include this in the line below
                self.layers[layer]['delta'] = (self.layers[layer]['activations'] - y_batch) #xent
                if optimiser == "SGD":
                    self.layers[layer]['weight_update'] = - self.lr * sum(self.weight_update((self.layers[layer]['activations'] - y_batch),  self.layers[layer-1]['activations']))     
                elif optimiser == "Adam":
                    self.layers[layer]['weight_update'] = - self.lr * self.Adam(self.layers[layer], self.layers[layer-1])
             #input      
            elif self.layers[layer]["layer_type"] == "input":
                self.layers[layer]['bias_update'] = - self.lr * sum(self.layers[layer+1]['delta'])  
            #hidden    
            else:
                self.layers[layer]['delta'] = self.cost_prime_hidden(self.layers[layer], self.layers[layer+1])                                        
                #get weight update
                if optimiser == "SGD":
                    self.layers[layer]['weight_update'] = - self.lr * self.SGD(self.layers[layer], self.layers[layer-1])
                elif optimiser == "Adam":
                    self.layers[layer]['weight_update'] = - self.lr * self.Adam(self.layers[layer], self.layers[layer-1])
                
                self.layers[layer]['bias_update'] = - self.lr *sum(self.layers[layer+1]['delta'])
                    
                    
        #adding weights and bias to their updates    
        for layer in range(len(self.layers)-1):
            if self.layers[layer]["layer_type"] == "input":
                self.layers[layer]['bias'] += self.layers[layer]['bias_update']
            elif self.layers[layer]['layer_type'] == "output":
                self.layers[layer]['weight_matrix'] +=self.layers[layer]['weight_update']
            else:
                self.layers[layer]['weight_matrix'] +=self.layers[layer]['weight_update']
                self.layers[layer]['bias'] += self.layers[layer]['bias_update']
    
    def calculate_activations(self, prev_layer, current_layer):
        product = np.matmul(prev_layer['activations'], current_layer['weight_matrix']) + prev_layer['bias']
        
        if current_layer['activation_func'] == "sigmoid":
            new_activations = self.sigmoid_activation_func(product)
        elif current_layer['activation_func'] == "relu":
            new_activations = self.relu_activation_func(product)
        elif current_layer['activation_func'] == "softmax":
            new_activations = self.softmax(product)
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
    
    def softmax(self, x):
        new_array = []
        for datapoint in x:
            stable_datapoint = datapoint - np.max(x)
            new_array.append(np.exp(stable_datapoint)/np.sum(np.exp((stable_datapoint))))
        return np.array(new_array)

    
    
    def Adam(self, current_layer, previous_layer, rho1 = 0.9, rho2 = 0.999, stab = 10e-8):
        gradient = self.SGD(current_layer, previous_layer)
        #I don't know why the t is here or what it does
        
        current_layer['s'] = rho1 * current_layer['s'] + (1 - rho1) * gradient
        current_layer['r'] = rho2 * current_layer['r'] + (1 - rho2) * gradient * gradient #operator between grad is hadamar/elementwise operator

        s_hat = current_layer['s'] / (1 - rho1)
        r_hat = current_layer['r'] / (1 - rho2)
        
        update = s_hat / (r_hat ** 0.5 + stab)
        
        return update
    
    def SGD(self, current_layer, prev_layer):
        weights = []
        for i in range(len(current_layer['delta'])):
            weights.append(current_layer['delta'][i] * prev_layer['activations'][i][:,np.newaxis])
        return sum(weights) / len(weights)

    
    def cost_prime_hidden(self, current_layer,layer_after):
        return np.array(self.activation_prime(current_layer)* np.matmul(layer_after['delta'], layer_after['weight_matrix'].transpose()))

    
    def weight_update(self, err, activation):
        weights = []
        for i in range(len(activation)):
            weights.append(err[i] * activation[i][:,np.newaxis])
        return weights
    
    
    def activation_prime(self, layer):
        if layer['activation_func'] == "sigmoid":
            return self.sigmoid_prime(layer['g_inputs'])
        elif layer['activation_func'] == 'relu':
            return self.relu_prime(layer['g_inputs'])

        
    def sigmoid_prime(self, g_input):
        return self.sigmoid_activation_func(g_input)*(1-self.sigmoid_activation_func(g_input)) 
    
    
    def relu_prime(self, g_input):
        return np.where(g_input>0, 1.0, 0.0)  
    
    
    def MSE(self, output_layer, y_batch):
        why = output_layer['activations'] - y_batch
        why_squared = why*why
        MSE = (1/len(y_batch))*np.sum(why_squared)
        return MSE
    
    
    def xent(self, output_layer, y_batch):
        return - sum(y_batch * np.log(output_layer))/len(y_batch)
    
    def accuracy_score(self,X, y):
        self.forward_pass(X)
        y_predicted = self.layers[-1]['activations']
        accuracy_sum = 0
        for datapoint in range(len(y)):
            if np.argmax(y[datapoint]) == np.argmax(y_predicted[datapoint]):
                accuracy_sum += 1
        
        accuracy = accuracy_sum/len(y)
        return accuracy*100
    
