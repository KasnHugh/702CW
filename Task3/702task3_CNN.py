# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:12:13 2020

@author: groes
"""
import load_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import json
from pytorchtools import EarlyStopping
from torch.utils.data.sampler import SubsetRandomSampler

######### Following: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/






# Setup a transform to apply to the MNIST data, and also the data set variables:
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# TO DO: 
    # Make kernel_size parameterizable 
    # Take this into account: If we wish to keep our input and output dimensions the same, with a filter size of 5 and a stride of 1, it turns out from the above formula that we need a padding of 2. Therefore, the argument for padding in Conv2d is 2.
    # Where  is the width of the input, F is the filter size, P is the padding and S is the stride. The same formula applies to the height calculation, but seeing as our image and filtering are symmetrical the same formula applies to both. If we wish to keep our input and output dimensions the same, with a filter size of 5 and a stride of 1, it turns out from the above formula that we need a padding of 2. Therefore, the argument for padding in Conv2d is 2.
    # implement some stopping criterion so that it is not the number of epocs that determines when the training stops
    # try out automated hyper parameter tuning
    # make regular feed forward net
    # implement drop out
    # use validation data? cf https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb 
class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        #  First, we create layer 1 (self.layer1) by creating a nn.Sequential object. 
        # This method allows us to create sequentially ordered layers in our network 
        # and is a handy way of creating a convolution + ReLU + pooling sequence.
        self.layer1 = nn.Sequential(
            # Conv2d creates a set of convolutional filters.
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def initialize_data(self, DATA_PATH, mean=0.1307, std=0.3081):
        # transforms to apply to the data
        # First argument converts the input data set to a PyTorch tensor.
        # 2nd argument normalizes data, because neural networks train better when the input data is normalized so that the data ranges from -1 to 1 or 0 to 1.
        # Note, that for each input channel a mean and standard deviation must be supplied – in the MNIST case, the input data is only single channeled
        transformer = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))]
            )

        # MNIST dataset
        self.train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=transformer, download=True)
        self.test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=transformer)
        print("Dataset saved to: " + DATA_PATH)
        
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        

        
    def train(self, num_epochs = 5, batch_size = 100, learning_rate = 0.001, DATA_PATH='./new_data', patience = 20):
        self.initialize_data(DATA_PATH) # where to save data
        self.batch_size = batch_size
        # Setup a transform to apply to the MNIST data, and also the data set variables:
        
        
        ### Adding this for validation and early stopping purposes
        # Using this guide: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb 
        # Creating indices for splitting training data into train and valid
        #validation_data_size = 0.2  
        #training_data_size = len(self.train_dataset)
        #indices = list(range(training_data_size))
        #np.random.shuffle(indices)
        #split = int(np.floor(validation_data_size * training_data_size))
        #training_idx, validation_idx = indices[split:], indices[:split]

        #train_sampler = SubsetRandomSampler(training_idx)
        #validation_sampler = SubsetRandomSampler(validation_idx)
        
        # WITHOUT EARLY STOPPING:
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        # WITH EARLY STOPPING:
        #train_loader = DataLoader(dataset=self.train_dataset,
        #                          batch_size=batch_size,
        #                          sampler=train_sampler,
        #                          num_workers=0)                
        #validation_loader = DataLoader(dataset=self.train_dataset,
        #                               batch_size=batch_size, 
        #                               sampler = validation_sampler,
        #                               num_workers=0)
        # Adding these to track losses while the model trains so that we can implement some stopping criteria
        # Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
        #validation_losses_per_forward_pass = [] # valid_losses
        #avg_validation_losses_all_epochs = [] # avg_valid_losses
        #early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        


        
        # Train the model      
        total_step = len(train_loader)
        #loss_list = [] # this was in the original guide, but its not being used
        #accuracy_list = [] # this was in the original guide, but its not being used
        
        
        # Implementing while loop with early stopping instead of for loop with epochs
        for epoch in range(num_epochs):
        
        # WITH EARLY STOPPING, USE THIS INSTEAD OF FOR LOOP
        #continue_training = True    
        #epoch = 0
        #while continue_training:
            #epoch += 1
            for i, (images, labels) in enumerate(train_loader):
                # Run the forward pass
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)
                #loss_list.append(loss.item()) # this was in the original guide, but its not being used
                
                print("Length of images:")
                print(len(images))
                print("Shape of images:")
                print(images.shape)
                print("Size of images:")
                print(images.size)
                print("Shape of labels:")
                print(labels.shape)   
                
                print("Shape of outputs")
                print(outputs.shape)
                
                # Backprop and perform Adam optimisation
                self.optimizer.zero_grad()
                ## OBS! It should not be necessary to call backward(). If this code works, try to get rid of backward
                loss.backward()
                self.optimizer.step()
        
                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                #accuracy_list.append(correct / total) # this was in the original guide, but its not being used
                
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))
        
            # WITH EARLY STOPPING, UNCOMMENT THIS:
            #self.eval()
            #with torch.no_grad():
            #    for images, labels in validation_loader:
            #        outputs = self.forward(images)
            #        loss = self.criterion(outputs, labels)
            #        validation_losses_per_forward_pass.append(loss.item())
            #avg_validation_loss_per_epoch = np.average(validation_losses_per_forward_pass)
            #avg_validation_losses_all_epochs.append(avg_validation_loss_per_epoch) 
            #validation_losses_per_forward_pass = []
            #early_stopping(avg_validation_loss_per_epoch, self)
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    continue_training = False
        
        
    def test(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Testing the model      
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                # train is being called when test() is run, so I'm amending the code below to see if it has any effects
                outputs = self.forward(images) # model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            test_accuracy = (correct / total) * 100
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(test_accuracy))
            return test_accuracy


        


###### Calling the model         
model = ConvNet1() 
model.train(num_epochs=1)
test_accuracy = model.test()
print(test_accuracy)

# Training and recording model performance
from RecordModelPerformance import RecordModelPerformance

settings = {"epochs" : [5, 5, 5, 5, 5, 5, 5, 5, 5],
            "batch_size" : [100, 100, 100, 10, 10, 10, 500, 500, 500],
            "learning_rate" : [0.001, 0.0001, 0.01, 0.001, 0.0001, 0.01, 0.001, 0.0001, 0.01]}

if len(settings["epochs"]) != len(settings["batch_size"]) != len(settings["learning_rate"]):
    raise Exception("Values in dict do not have same length")
    
model = ConvNet1() 
results = {}
models_trained = 0
models_to_train = len(settings["epochs"])

for i in range(len(settings["epochs"])):

    epochs = settings["epochs"][i]
    batch_size = settings["batch_size"][i] 
    learning_rate = settings["learning_rate"][i]
    recorder = RecordModelPerformance(model_object=model, result_dict=results,
                                 model_name="ConvNet1", epochs=epochs, batch_size=batch_size,
                                 learning_rate=learning_rate)
    results = recorder.run()
    print("Models trained: {} out of {}".format(models_trained, models_to_train))
    models_trained += 1

# Saving dict as pd df
results_df = pd.DataFrame.from_dict(data=results)
results_df.to_excel("results_df.xlsx")

# Saving dict as json
dictionary_name = "ConvNet1_results.json"
json = json.dumps(results)
f = open(dictionary_name,"w")
f.write(json)
f.close()
    








#model.train(num_epochs = 1)
#test_accuracy = model.test()

# Hyperparameters
#num_epochs = 5
#num_classes = 10
#batch_size = 100
#learning_rate = 0.001
DATA_PATH = './data'
MODEL_STORE_PATH = './models'


# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')     
        


        
        
        
        
        