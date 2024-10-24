# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]

#Compute the cost function
def cost(Y_hat, Y):
    n = Y.shape[0]
    c = -np.sum(Y*np.log(Y_hat)) / n
    
    return c

def test(Y_hat, Y):
    Y_out = np.zeros_like(Y)
    
    idx = np.argmax(Y_hat[-1], axis=1)
    Y_out[range(Y.shape[0]),idx] = 1
    acc = np.sum(Y_out*Y) / Y.shape[0]
    print("Training accuracy is: %f" %(acc))
      
    return acc

###### Training loops
###### Learning rate
###### The number of layers
###### The number of convolutional kernels in each layer
###### The size of convolutional kernels
###### The size of pooling kernels

data = np.load("data.npy")

X = data[:,:-1].reshape(data.shape[0], 20, 20).transpose(0,2,1)
Y = data[:,-1].astype(np.int32)
(n, L, _) = X.shape
Y = onehotEncoder(Y, 10)

test(Y_hat, Y)