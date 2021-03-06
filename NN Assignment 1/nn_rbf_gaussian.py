# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 19:54:17 2021

@author: Shiva
"""

from scipy.io import loadmat
#import pandas as pd
import numpy as np
import math
import random
from scipy.spatial import distance

data_test = loadmat('data_test.mat')
data_train = loadmat('data_train.mat')
label_train = loadmat('label_train.mat')
db_data_train = data_train["data_train"]
db_label_train = label_train["label_train"]
db_data_test = data_test["data_test"]
num_centres = 200
db_weight = np.zeros([num_centres,1])
centres = np.zeros([num_centres,33])
index_seed = random.sample(range(0,329), num_centres)
#print(index_seed)
for i in range(num_centres):
    centres[i,:] = db_data_train[index_seed[i]]
    
#Calculate Sigma based on centres. sigma = dmax/sqrt(2m). m is number of centres. dmax is max distance between centres
d_max = 0
for i in range(num_centres):
    for j in range(num_centres):
        if distance.euclidean(centres[i], centres[j]) > d_max:
            d_max = distance.euclidean(centres[i], centres[j])

#print(db_data_train)
#print(db_data_test)
#print(db_label_train)
#print(db_weight)

#For Only 2 centres. Each Center for each Class
db_class1 = np.zeros([1,33])
db_class2 = np.zeros([1,33])
num_class_1 = 0
num_class_2 = 0

for i in range(len(db_data_train)):
    if(db_label_train[i] == 1):
        db_class1 += db_data_train[i,:]
        num_class_1 += 1
    elif(db_label_train[i] == -1):
        db_class2 += db_data_train[i,:]
        num_class_2 += 1

db_avg1 = db_class1/num_class_1
db_avg2 = db_class2/num_class_2

#Calculate Sigma based on centres
ligma = d_max/math.sqrt(2*num_centres)

def fi(x,c,sigma):
    db_phi = np.zeros([len(x),len(c)])
    for i in range(len(x)):
        for j in range(len(c)):
            xcj = np.linalg.norm(x[i]-c[j])
            db_phi[i,j] = math.exp(-(xcj*xcj)/(2*sigma*sigma))
            
    return db_phi

db_fi = fi(db_data_train, centres, ligma)
db_weight = np.matmul(np.linalg.pinv(db_fi),db_label_train)

#print(db_weight)

def acc(x,lablez, centres, sigma, w):
    num_corr = 0
    predicts = np.matmul(fi(x, centres, sigma), w) 
    threshold = np.average(np.unique(lablez))
    for i in range(len(predicts)):
      if predicts[i] > threshold:
        predicts[i] = 1
      else:
        predicts[i] = -1
    for i in range(len(predicts)):
      if (predicts[i] == lablez[i]):
        num_corr +=1 
    acc = num_corr/len(x)
    return acc

print(acc(db_data_train,db_label_train, centres, ligma, db_weight))

prediction = np.matmul(fi(db_data_test, centres, ligma), db_weight)
for i in range(len(prediction)):
      if prediction[i] > 0:
        prediction[i] = 1
      else:
        prediction[i] = -1
print(prediction.astype(int))