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
    Computes mean squared error gradient between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    
    return 2*(outi-out0)/outi.shape[1]

@njit(cache=True,fastmath=False, parallel=True)
def MAE_loss(predictions, targets):
    """
    Computes Mean Absolute error/loss between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    # Alternative 1
    # return np.sum(np.abs(predictions-targets))/predictions.shape[1]
    # Alternative 2
    # return np.mean(np.abs(predictions - targets))*predictions.shape[0]
    # Alternative 3 (Extremely fast if there are a lot of cores available)
    loss = 0.0
    for i in prange(predictions.shape[0]):
        for j in prange(predictions.shape[1]):
            loss = loss + np.abs(predictions[i,j] - targets[i,j]) # should have given a race condition but somehow numba is able to avoid it
    # Average over number of output nodes
    loss = loss / predictions.shape[1]
    return loss


@njit(cache=True,fastmath=True,parallel=False)
def MAE_loss_grad(predictions, targets):
    """
    Computes mean absolute error gradient between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """ 
    # References: https://stats.stackexchange.com/questions/313235/can-i-use-gradient-descent-for-least-absolute-deviation-regression
    # https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative
    # Alternative 1
    loss_grad = np.where(predictions >= targets, 1.0, -1.0)
    return loss_grad/predictions.shape[1]
    # Alternative 2 (The following is wrong (only returns a scalar). Needs to be corrected (return an (N,k) array). )
    loss_grad = 0.0
    for i in prange(predictions.shape[0]):
        for j in prange(predictions.shape[1]):
            # if np.abs(predictions[i,j] - targets[i,j])<0.0000001:
            #     loss_grad = loss_grad + 0.0 # should have given a race condition but somehow numba is able to avoid it
            if predictions[i,j] < targets[i,j]:
                loss_grad = loss_grad + (-1.0) # should have given a race condition but somehow numba is able to avoid it
            elif predictions[i,j] > targets[i,j]: 
                loss_grad = loss_grad + (1.0) # should have given a race condition but somehow numba is able to avoid it
    # Average over number of output nodes
    loss_grad = loss_grad / predictions.shape[1]
    return loss_grad

@njit(cache=True,fastmath=True, parallel=True)
def BCE_loss(predictions, targets, epsilon=1e-7):
    """
    Computes binary cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    # Reference Formula: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy
    # This should be averaged over outputs/output nodes/ classes
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
    # Alternative 1 (This is probably as fast as the other one)
    # ce = -np.sum(targets*np.log(predictions+epsilon)+(1.-targets)*np.log(1.-predictions+epsilon))/predictions.shape[1]#/N
    # return ce
    # Alternative 2 (Parallelized) 
    loss = 0.0
    for i in prange(predictions.shape[0]):
        for j in prange(predictions.shape[1]):
            temp = -(targets[i,j]*np.log(predictions[i,j]+epsilon) + (1.-targets[i,j])*np.log(1.-predictions[i,j]+epsilon))
            # temp = -(targets[i,j]*np.log(predictions[i,j]) + (1.-targets[i,j])*np.log(1.-predictions[i,j]))
            loss = loss + temp # should have given a race condition but somehow numba is able to avoid it
    # Average over number of output nodes
    loss = loss / predictions.shape[1]
    return loss

# @njit(cache=True,fastmath=True, parallel=True)
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
    # Alternative 1 (Converts NAN resulting from division by 0 to 0 using Numpy, is incompatible with NUMBA)
    return -(np.nan_to_num(np.divide(targets,predictions,dtype=targets.dtype))-np.nan_to_num(np.divide(1.-targets,1.-predictions,dtype=targets.dtype)))/predictions.shape[1]
    # Alternative 2 (Converts NAN resulting from division by 0 to 0 using Numpy, is compatible with NUMBA) !!! DOESNT WORK
    # grad = np.zeros(predictions.shape)
    # # grad = predictions
    # for i in prange(predictions.shape[0]):
    #     for j in prange(predictions.shape[1]):
    #         temp = targets[i,j]/predictions[i,j]  -  (1.-targets)/(1.-predictions[i,j])
    #         if not np.isnan(temp):
    #             grad[i,j] = temp
    # # Average over number of output nodes
    # grad = grad / predictions.shape[1]
    # return grad
    # Alternative 3 !! DOESNT WORK
    # grad = -(np.divide(targets,predictions)-np.divide(1.-targets,1.-predictions))/predictions.shape[1]
    # for i in prange(predictions.shape[0]):
    #     for j in prange(predictions.shape[1]):
    #         if np.isnan(grad[i,j]):
    #             grad[i,j]=0.
    # return grad

@njit(cache=True,fastmath=False)
def CCE_loss(predictions, targets, epsilon=1e-12):
    """
    Computes categorical cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: scalar
    """
    # Reference: https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
    # Formula: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy
    # No averaging over the no. of outputs/output nodes/classes is needed
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    cce = -np.sum(targets*np.log(predictions+1e-9))
    return cce

# @njit(cache=True,fastmath=True)
def CCE_loss_grad(predictions, targets):
    """
    Computes categorical cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: matrix
    """
    # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    return -np.nan_to_num(np.divide(targets,predictions,dtype=targets.dtype))
    # return -np.nan_to_num(np.divide(targets,predictions))
    # return -np.divide(targets,predictions)

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
    Computes mean squared error gradient between targets 
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    
    return 2*(outi-out0)/outi.shape[1]

def MAE_loss_cupy(predictions, targets):
    """
    Computes Mean Absolute error/loss between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    # Alternative 1
    return cp.sum(cp.abs(predictions-targets))/predictions.shape[1]
    # Alternative 2
    # return cp.mean(cp.abs(predictions - targets))*predictions.shape[0]
    


def MAE_loss_grad_cupy(predictions, targets):
    """
    Computes mean absolute error gradient between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
          targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """ 
    # References: https://stats.stackexchange.com/questions/313235/can-i-use-gradient-descent-for-least-absolute-deviation-regression
    # https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative
    # Alternative 1
    loss_grad = cp.where(predictions >= targets, 1.0, -1.0)
    return loss_grad/predictions.shape[1]
    

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

def CCE_loss_cupy(predictions, targets, epsilon=1e-7):
    """
    Computes categorical cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: scalar
    """
    # Reference: https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
    # Formula: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy
    # No averaging over the no. of outputs/output nodes/classes is needed
    predictions = cp.clip(predictions, epsilon, 1. - epsilon)
    cce = -cp.sum(targets*cp.log(predictions))
    return cce

def CCE_loss_grad_cupy(predictions, targets):
    """
    Computes cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: matrix
    """
    # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    return -cp.divide(targets,predictions)