'''
Activation functions and their gradients
crysx_nn 
Manas Sharma
feedback@bragitoff.com
https://www.bragitoff.com
'''
from numba import vectorize,jit,njit,prange,set_num_threads,get_num_threads 
import numpy as np


@njit(cache=True,fastmath=True)  # Best implementation (VERY FAST)
def Softmax(x):
    '''
    Performs the softmax activation on a given set of inputs
    Input: x (N,k) ndarray (N: no. of samples, k: no. of nodes)
    Returns: 
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)

    '''
    e_x = np.exp(x - np.max(x)) # For stability
#     return e_x / e_x.sum(axis=1, keepdims=True) # only difference
    return e_x / e_x.sum(axis=1).reshape((-1, 1)) # Alternative of keepdims=True for Numba compatibility

@njit(cache=True,fastmath=True)
def Softmax_grad(x): # Best implementation (VERY FAST)
    '''Returns the jacobian of the Softmax function for the given set of inputs.
    Inputs:
    x: should be a 2d array where the rows correspond to the samples
        and the columns correspond to the nodes.
    Returns: jacobian
    '''
    s = Softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=np.float32)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for k in range(s.shape[1]):
                temp1[i,j,k] = s[i,j]*a[j,k]
                temp2[i,j,k] = s[i,j]*s[i,k]
    return temp1-temp2

@njit(cache=True,fastmath=True)
def Sigmoid(x):
    return 1/(1+np.exp(-x))

@njit(cache=True,fastmath=True)
def Sigmoid_grad(x):
    return np.exp(-x)/(np.exp(-x)+1)**2

@njit(cache=True,fastmath=True)
def ReLU(x):
    return np.max(0,x)

@njit(cache=True,fastmath=True)
def ReLU_grad(x):
    return np.max(0,x)

@njit(cache=True,fastmath=True)
def Tanh_offset(x):
    return 0.5*(1+np.tanh(x))

@njit(cache=True,fastmath=True)
def Tanh_offset_grad(x):
    return 1/(np.cosh(2*x)+1)