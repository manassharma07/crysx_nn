'''
Loss functions and their gradients
crysx_nn 
Manas Sharma
feedback@bragitoff.com
https://www.bragitoff.com
'''
from numba import vectorize,jit,njit,prange,set_num_threads,get_num_threads 
import numpy as np

@njit(cache=True,fastmath=True)
def MSE_loss(outi, out0):
    """
    Computes Mean Squared error/loss between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    
    return np.sum((outi-out0)**2)/outi.shape[1]

@njit(cache=True,fastmath=True)
def MSE_loss_grad(outi, out0):
    """
    Computes mean squared error gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    
    return 2*(outi-out0)/outi.shape[1]

@njit(cache=True,fastmath=True)
def BCE_loss(predictions, targets, epsilon=1e-12):
    """
    Computes binary cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+epsilon)+(1-targets)*np.log(1-predictions+epsilon))/predictions.shape[1]#/N
    return ce

@njit(cache=True,fastmath=True)
def BCE_loss_grad(predictions, targets):
    """
    Computes binary cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    return -(np.divide(targets,predictions)-np.divide(1-targets,1-predictions))/predictions.shape[1]