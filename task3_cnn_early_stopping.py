# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:42:38 2020

@author: groes
"""

"""

CNN with early stopping

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
from pytorchtools import EarlyStopping
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

######### Following: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/






# Setup a transform to apply to the MNIST data, and also the data set variables:
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# TO DO: 
    # Take this into account: If we wish to keep our input and output dimensions the same, with a filter size of 5 and a stride of 1, it turns out from the above formula that we need a padding of 2. Therefore, the argument for padding in Conv2d is 2.
    # Where  is the width of the input, F is the filter size, P is the padding and S is the stride. The same formula applies to the height calculation, but seeing as our image and filtering are symmetrical the same formula applies to both. If we wish to keep our input and output dimensions the same, with a filter size of 5 and a stride of 1, it turns out from the above formula that we need a padding of 2. Therefore, the argument for padding in Conv2d is 2.
    # implement some stopping criterion so that it is not the number of epocs that determines when the training stops
    # try out automated hyper parameter tuning
    # use validation data? cf https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb 


def create_datasets(batch_size):
    # Adapted from: 
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb

    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    train_data = datasets.MNIST(root='cnn2data', 
                                train=True,
                                download=True, 
                                transform=transform)

    test_data = datasets.MNIST(root='cnn2data',
                               train=False,
                               download=True,
                               transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)
    
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=0)
    
    return train_loader, test_loader, valid_loader


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
        
    #def initialize_data(self, DATA_PATH, mean=0.1307, std=0.3081):
        # transforms to apply to the data
        # First argument converts the input data set to a PyTorch tensor.
        # 2nd argument normalizes data, because neural networks train better when the input data is normalized so that the data ranges from -1 to 1 or 0 to 1.
        # Note, that for each input channel a mean and standard deviation must be supplied – in the MNIST case, the input data is only single channeled
     #   transformer = transforms.Compose(
      #      [transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))]
      #      )

        # MNIST dataset
       # self.train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=transformer, download=True)
       # self.test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=transformer)
       # print("Dataset saved to: " + DATA_PATH)
        
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
    
def train(model, train_loader, valid_loader, num_epochs = 5,
          batch_size = 100, learning_rate = 0.001, patience = 20):
    
    model.criterion = nn.CrossEntropyLoss()

    # specify optimizer
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #self.initialize_data(DATA_PATH) # where to save data
    model.batch_size = batch_size
    
    ### WITH EARLY STOPPING
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, num_epochs + 1):
        
        for i, (images, labels) in enumerate(train_loader):
            
            ## TRAINING ##
            # Run the forward pass
            outputs = model.forward(images)
            loss = model.criterion(outputs, labels)
            
            # Backprop and perform Adam optimisation
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            train_losses.append(loss.item())
            
            ## VALIDATING ##
            model.eval()
            for data, labels in valid_loader:
                output = model.forward(data)
                loss = model.criterion(output, labels)
                valid_losses.append(loss.item())
        
        # Model evaluation source: 
        # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb 
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses #model, avg_train_losses, avg_valid_losses
                
            
        
    
    
    # WITHOUT EARLY STOPPING:
    #train_loader = DataLoader(dataset=training_data,
    #                          batch_size=batch_size,
    #                          shuffle=True)
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
    """
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
            
            #print("Length of images:")
            #print(len(images))
            #print("Shape of images:")
            #print(images.shape)
            #print("Size of images:")
            #print(images.size)
            #print("Shape of labels:")
            #print(labels.shape)   
            
            #print("Shape of outputs")
            #print(outputs.shape)
            
            # Backprop and perform Adam optimisation
            self.optimizer.zero_grad()
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



def test(model, test_loader):
        print("Starting test")
        #test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)
        print("Test loader created")
        # Testing the model      
        #self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                print("Running batch")
                # train is being called when test() is run, so I'm amending the code below to see if it has any effects
                outputs = model.forward(images) # model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            test_accuracy = (correct / total) * 100
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(test_accuracy))
            return test_accuracy
 """

def test(model, test_loader):
        print("Starting test")
        #test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)
        print("Test loader created")
        # Testing the model      
        #self.eval()
        all_images = []
        all_labels = []
        all_predicted = []
        all_targets = []
        
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                all_images.append(images)
                all_labels.append(labels)
                print("Running batch")
                # train is being called when test() is run, so I'm amending the code below to see if it has any effects
                outputs = model.forward(images) # model(images)
                target, predicted = torch.max(outputs.data, 1)
                all_targets.append(target)
                all_predicted.append(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            test_accuracy = (correct / total) * 100
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(test_accuracy))
            return test_accuracy, all_predicted, all_labels
        



def get_confusion_matrix(model, test_loader, num_classes):
    # Adapted from 
    # https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial


    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_loader):
            #inputs = inputs.to(device)
            #classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    
    print(confusion_matrix)

###### Calling the model

model = ConvNet1() 

batch_size = 256
n_epochs = 100
patience = 20
activation_func = "relu"
learningrate = 0.001
train_loader, test_loader, valid_loader = create_datasets(batch_size)

model, train_loss, valid_loss = train(model,
    train_loader=train_loader, valid_loader=valid_loader, num_epochs = n_epochs,
    batch_size=batch_size, learning_rate=learningrate, patience=patience) #train_loader, valid_loader, n_epochs, batch_size, learningrate, patience)

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')

### TESTING THE MODEL
test_accuracy, all_predicted, all_labels = test(model, test_loader)
experiment_no = 10

     
model_specs = {
    "Experiment" : str(experiment_no),
    "Test_accuracy" : [test_accuracy],
    "Patience" : [patience],
    "Epochs" : [n_epochs],
    "Batch_size" : [batch_size],
    "Model_spec" : [model],
    "Activation_func" : [activation_func],
    "Optimizer" : [model.optimizer],
    "Loss func" : [model.criterion]
    
    }


results_df = pd.DataFrame.from_dict(data=model_specs)

#experiments = ["1", "2", "3", "4", "5"]

all_results = all_results.append(results_df)
experiment_no += 1 

#all_results.insert(0, "Experiments", experiments)

all_results.to_excel("all_results_task3.xlsx")


##### MAKING CONFUSION MATRIX
cm_aggregated = np.zeros((10,10)).astype(int)
for pred, label in zip(all_predicted, all_labels):
    cm = confusion_matrix(pred, label)
    cm_aggregated += cm

cm_as_df = pd.DataFrame(data = cm_aggregated)

cm_as_df.to_excel("confusion_matrix_best_model.xlsx")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plot_confusion_matrix(cm_aggregated, [0,1,2,3,4,5,6,7,8,9])

##############################################################################
"""
#mean=0.1307
#std=0.3081
#DATA_PATH='./new_data'  
#transformer = transforms.Compose([transforms.ToTensor(),
#                                  transforms.Normalize(mean=(mean,), std=(std,))])   

#train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=transformer, download=True)
#test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=transformer)
model = ConvNet1() 
model.train(train_dataset, num_epochs=1)
test_accuracy = model.test(test_dataset)
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
    

"""
    
    
    
    
    