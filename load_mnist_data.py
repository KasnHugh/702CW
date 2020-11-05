# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:55:26 2020

@author: groes
"""

def load_data(path, dl):
    '''
    Parameters:
    
    path (str): The path to the folder you either want to download the dataset to
    or where the test.pt and training.pt files are located if you have already downloaded.
    
    dl (boolean): If True, the dataset is downloaded to the path, if False, the function
    uses the test.pt and training.pt files already downloaded
    
    Returns:
    The mnist dataset
    
    See also: https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
    
    '''
    mnist_dataset = dset.MNIST(
        root = path,
        download = True
        )
    return mnist_dataset


