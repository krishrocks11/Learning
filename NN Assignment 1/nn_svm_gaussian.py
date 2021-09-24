# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:39:00 2021

@author: Shiva
"""

#from enum import auto
#import numpy as np
#from numpy.core.numeric import correlate
###import the scipy for loading .mat matlab data
from scipy.io import loadmat
### loading file and taking out data and label
data_test = loadmat('data_test.mat')
data_train = loadmat('data_train.mat')
label_train = loadmat('label_train.mat')
label_test = loadmat('label_test.mat')
db_data_train = data_train["data_train"]
db_label_train = label_train["label_train"]
db_data_test = data_test["data_test"]
db_label_test = label_test["label_test"]

#### printing the type and size of the variables
#tbd


X_train = db_data_train
y_train = db_label_train
X_test = db_data_test
y_test = db_label_test

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', gamma=0.03)
svclassifier.fit(X_train, y_train)
#To use Gaussian kernel, you have to specify 'rbf' as value for the Kernel parameter of the SVC class.

#Prediction and Evaluation
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#The output of the Kernel SVM with Gaussian kernel looks like this:
