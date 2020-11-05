# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:55:26 2020

@author: groes
"""

def load_data(path):
    '''
    Parameters
    ----------
    path : str
        Absolute path to the folder you either want to store the dataset or already have it
        
    Returns
    -------
    mnist_dataset.train_data, mnist_dataset.test_data : mnist objects
        the training and testing data of the mnist dataset

    '''
    import torchvision.datasets as dset
    mnist_dataset = dset.MNIST(
        root = path,
        download = True
        )
    
    return mnist_dataset.train_data, mnist_dataset.test_data
