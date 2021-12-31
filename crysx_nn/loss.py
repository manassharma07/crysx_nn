'''
Loss functions and their gradients
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

@njit(cache=True,fastmath=False, parallel=True)
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
    #TODO : There is big problem that I noticed today. For large arrays, the NUmba compiled version of this 
    # doesn't give correct results.
    # but then the pure numpy version is not as fast as Numba. 
    # Alternative 1
    # return np.sum((outi-out0)**2)/outi.shape[1]
    # Alternative 2
    # return np.mean((outi - out0)**2)*outi.shape[0]
    # Alternative 3 (Extremely fast if there are a lot of cores available)
    loss = 0.0
    for i in prange(outi.shape[0]):
        for j in prange(outi.shape[1]):
            loss = loss + (outi[i,j] - out0[i,j])**2 # should have given a race condition but somehow numba is able to avoid it
    # Average over number of output nodes
    loss = loss / outi.shape[1]
    return loss


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

@njit(cache=True,fastmath=False, parallel=True)
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
    # Alternative 1
    # ce = -np.sum(targets*np.log(predictions+epsilon)+(1-targets)*np.log(1-predictions+epsilon))/predictions.shape[1]#/N
    # return ce
    # Alternative 2 (Parallelized)
    loss = 0.0
    for i in prange(predictions.shape[0]):
        for j in prange(predictions.shape[1]):
            temp = -(targets[i,j]*np.log(predictions[i,j]+epsilon) + (1.-targets[i,j])*np.log(1.-predictions[i,j]+epsilon))
            loss = loss + temp # should have given a race condition but somehow numba is able to avoid it
    # Average over number of output nodes
    loss = loss / predictions.shape[1]
    return loss

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
    # Alternative 1
    return -(np.divide(targets,predictions)-np.divide(1-targets,1-predictions))/predictions.shape[1]


##-----------CUPY---------------
def MSE_loss_cupy(outi, out0):
    """
    Computes Mean Squared error/loss between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    
    return cp.sum((outi-out0)**2)/outi.shape[1]


def MSE_loss_grad_cupy(outi, out0):
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

def BCE_loss_cupy(predictions, targets, epsilon=1e-12):
    """
    Computes binary cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    predictions = cp.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
    ce = -cp.sum(targets*cp.log(predictions+epsilon)+(1-targets)*cp.log(1-predictions+epsilon))/predictions.shape[1]#/N
    return ce


def BCE_loss_grad_cupy(predictions, targets):
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
    return -(cp.divide(targets,predictions)-cp.divide(1-targets,1-predictions))/predictions.shape[1]