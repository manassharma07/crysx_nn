'''
Activation functions and their gradients
crysx_nn 
Manas Sharma
feedback@bragitoff.com
https://www.bragitoff.com
'''
from numba import vectorize,jit,njit,prange,set_num_threads,get_num_threads 
import numpy as np
try:
    import cupy as cp                     
except ImportError:
    print('CuPy could not be imported!')



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
    # Einsum is unsupported with Numba (nopython mode)
    # temp1 = np.einsum('ij,jk->ijk',s,a)
    # temp2 = np.einsum('ij,ik->ijk',s,s)
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
    # Broadcasting seems expensive compared to TF and PyTorch
    # return np.maximum(0.,x)
    a = np.zeros(x.shape,dtype=np.float32)
    return np.maximum(a,x)

@njit(cache=True,fastmath=True)
def ReLU_grad(x):
    return np.greater(x, 0.).astype(np.float32)

@njit(cache=True,fastmath=True)
def Tanh_offset(x):
    return 0.5*(1+np.tanh(x))

@njit(cache=True,fastmath=True)
def Tanh_offset_grad(x):
    return 1/(np.cosh(2*x)+1)







###-----------CUPY----------------
def Softmax_cupy(x):
    '''
    Performs the softmax activation on a given set of inputs
    Input: x (N,k) ndarray (N: no. of samples, k: no. of nodes)
    Returns: 
    Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
    '''
    e_x = cp.exp(x - cp.max(x)) # For stability
#     return e_x / e_x.sum(axis=1, keepdims=True) # only difference
    return e_x / e_x.sum(axis=1).reshape((-1, 1)) # Alternative of keepdims=True for Numba compatibility

def Softmax_grad_cupy(x): # Best implementation (VERY FAST)
    '''Returns the jacobian of the Softmax function for the given set of inputs.
    Inputs:
    x: should be a 2d array where the rows correspond to the samples
        and the columns correspond to the nodes.
    Returns: jacobian
    '''
    s = Softmax_cupy(x)
    a = cp.eye(s.shape[-1])
    temp1 = cp.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=cp.float32)
    temp2 = cp.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=cp.float32)
    # for i in range(s.shape[0]):
    #     for j in range(s.shape[1]):
    #         for k in range(s.shape[1]):
    #             temp1[i,j,k] = s[i,j]*a[j,k]
    #             temp2[i,j,k] = s[i,j]*s[i,k]
    temp1 = cp.einsum('ij,jk->ijk',s,a)
    temp2 = cp.einsum('ij,ik->ijk',s,s)
    return temp1-temp2

def Sigmoid_cupy(x):
    return 1/(1+cp.exp(-x))

def Sigmoid_grad_cupy(x):
    return cp.exp(-x)/(cp.exp(-x)+1)**2


def ReLU_cupy(x):
    # Broadcasting seems expensive compared to TF and PyTorch
    # return np.maximum(0.,x)
    a = cp.zeros(x.shape,dtype=cp.float32)
    return cp.maximum(a,x)

def ReLU_grad_cupy(x):
    return cp.greater(x, 0.).astype(cp.float32)