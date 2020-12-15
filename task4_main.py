# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:22:59 2020

@author: groes
"""
import task4_data as t4
#import neural_network as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


df = pd.read_csv("df_all_data_w_desc_2020-12-14.csv")

df = t4.enrich_dataset(df)

#training_data, validation_data = t4.add_training_validation_set(df, 0.8)

