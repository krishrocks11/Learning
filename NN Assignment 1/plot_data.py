# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:54:35 2021

@author: Shiva
"""
###import the scipy for loading .mat matlab data
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
### loading file and taking out data and label
data_test = loadmat('data_test.mat')
data_train = loadmat('data_train.mat')
label_train = loadmat('label_train.mat')
label_test = loadmat('label_test.mat')
db_data_train = data_train["data_train"]
db_label_train = label_train["label_train"]
db_data_test = data_test["data_test"]
db_label_test = label_train["label_test"]

#y = np.linspace(1,21,1)
print(np.mean(db_data_test, axis=1))
y = db_data_test[:,1]
plt.scatter(db_data_test[:,0], y)
plt.show()