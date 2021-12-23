import numpy as np
# print(np.__version__)
from autograd import numpy as anp
from autograd import grad, elementwise_grad, jacobian
if hasattr(__builtins__,'__IPYTHON__'):
    from tqdm.notebook import tqdm
    # print('Using notebook tqdm')
else:
    from tqdm import tqdm
    # print('Using simple tqdm')

from tqdm.auto import tqdm
import numexpr as ne
from opt_einsum import contract, contract_expression
from numba import vectorize,jit,njit,prange,set_num_threads,get_num_threads 
from nnv import NNV
import matplotlib.pyplot as plt
from operator import add

# act_func_dict = {'Sigmoid':Sigmoid,'ReLU':ReLU,'ELU':ELU, 'Hardshrink' : Hardshrink,'Hardsigmoid':Hardsigmoid,\
#                  'Hardtanh':Hardtanh,'Hardswish':Hardswish,'LeakyReLU':LeakyReLU,'LogSigmoid':LogSigmoid,\
#                  'Tanh':Tanh,'Softmax':Softmax,'Softmin':Softmin,'LogSoftmax':LogSoftmax,'Mish':Mish,\
#                  'Swish':Swish,'Exponential':Exponential}





def visualize(nInputs, neurons_per_layer, activation_func_names):
    plt.rcParams["figure.figsize"] = (14,10)
    nLayers = len(neurons_per_layer)
    # Input layer
    layersList = [{"title":"input"+"\n(n="+str(nInputs)+")", "units": nInputs, "color": "darkBlue"}]
    # Hidden layers
    if nLayers>1:
        for i in range(nLayers-1):
            layersList.append({"title":"hidden "+str(i+1)+"\n("+activation_func_names[i]+")"+"\n(n="+str(neurons_per_layer[i])+")", "units": neurons_per_layer[i]})
        

    # Output layers
    layersList.append({"title":"output \n("+activation_func_names[nLayers-1]+")"+"\n(n="+str(neurons_per_layer[nLayers-1])+")", "units": neurons_per_layer[nLayers-1]})

    NNV(layersList,max_num_nodes_visible=10, node_radius=5, spacing_layer=60, font_size=17).render()
    plt.rcParams["figure.figsize"] = (9,6)
    plt.show()
    

@njit(cache=True,fastmath=True)
def tanhOffsetAct(x):
    return 0.5*(1+np.tanh(x))

@njit(cache=True,fastmath=True)
def Sigmoid(x):
    return 1/(1+np.exp(-x))
#     return ne.evaluate('1/(1+exp(x))')

@njit(cache=True,fastmath=True)
def ReLU(x):
    return np.max(0,x)

@njit(cache=True,fastmath=True)
def Softmax1(x):
    """Compute the softmax of vector x."""
#     return np.exp(x) / np.sum(np.exp(x), axis=0)
#     More stable
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

# correct solution:
@njit(cache=True,fastmath=True)  # Best implementation (VERY FAST)
def Softmax(x):
    """Works for 2D arrays only(rows for samples, columns for nodes/outputs)"""
    e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=1, keepdims=True) # only difference
    return e_x / e_x.sum(axis=1).reshape((-1, 1)) # Alternative of keepdims=True for Numba compatible

def SoftmaxAutoGrad(x):
    """Works for 2D arrays only(rows for samples, columns for outputs)"""
    e_x = anp.exp(x - anp.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True) # only difference

# my (correct) solution:
def Softmax2(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def Stable_softmax(x):
    '''Works for both 1d and 2d arrays'''
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax





@njit(cache=True,fastmath=True)
def errorFunc(outi, out0):
    '''
    Error function
    out0 is the expected output (exact/ideal)
    outi is the calculated output
    '''
    
    return np.sum((outi-out0)**2)/outi.shape[1]

def errorFunc_(outi, out0):
    '''
    Error function
    out0 is the expected output (exact/ideal)
    outi is the calculated output
    '''
    
    return anp.sum((outi-out0)**2)/outi.shape[1]

@njit(cache=True,fastmath=True)
def errorFuncGrad(outi, out0):
    '''
    Error function
    out0 is the expected output (exact/ideal)
    outi is the calculated output
    returns matrix
    '''
    
    return 2*(outi-out0)/outi.shape[1]

gradErrorFunc = errorFuncGrad

@njit(cache=True,fastmath=True)
def tanhOffsetActFuncPrime(x):
    return 1/(np.cosh(2*x)+1)

@njit(cache=True,fastmath=True)
def Sigmoid_grad(x):
    return np.exp(-x)/(np.exp(-x)+1)**2
#     return ne.evaluate('exp(-x)/(exp(-x)+1)**2')

@njit(cache=True,fastmath=True)
def ReLU_grad(x):
    return np.max(0,x)

@njit(cache=True,fastmath=True)
def ReLUdActFuncPrime(x):
    return np.greater(z, 0).astype(int)
#     return return np.heaviside(x, 0)
#     return  (x > 0) * 1

@njit(cache=True,fastmath=True)
def SoftmaxActFuncPrime(x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# @njit(cache=False,fastmath=True)
def SoftmaxActFuncPrime1(x):
    x = Softmax(x)
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# @njit(cache=False,fastmath=True)    
def Softmax_dash(x):
    '''Expects input to be an Nx1 2d array'''
    I = np.eye(x.shape[0])
    return Softmax(x) * (I - Softmax(x).T)

def SoftmaxActFuncPrime(x):
    signal = Softmax(x)
    J = - signal[..., None] * signal[:, None, :] # off-diagonal Jacobian
    iy, ix = np.diag_indices_from(J[0])
    J[:, iy, ix] = signal * (1. - signal) # diagonal
    return J.sum(axis=1) # sum across-rows for each sample

# @njit(cache=False,fastmath=True)  
def Softmax_d(x):
#     return softmax_broadcast(z) * (1 - softmax_broadcast(z))
    s = Softmax(x)
    return s * (1 - s)

def cross_ent(y, yhat):
    return -1 * np.sum(y * np.log(yhat+1e-16))

def new(a):
    return a


@njit(cache=True,fastmath=True)
def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/predictions.shape[1]#/N
    return ce

@njit(cache=True,fastmath=True)
def binary_cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+epsilon)+(1-targets)*np.log(1-predictions+epsilon))/predictions.shape[1]#/N
    return ce

@njit(cache=True,fastmath=True)
def binary_cross_entropy_grad(predictions, targets):
    """
    Computes cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: matrix
    """
    # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    return -(np.divide(targets,predictions)-np.divide(1-targets,1-predictions))/predictions.shape[1]

@njit(cache=True,fastmath=True)
def cross_entropy_grad(predictions, targets):
    """
    Computes cross entropy gradient between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: matrix
    """
    # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    return -np.divide(targets,predictions)

def cross_entropy2(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
          targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = anp.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
    ce = -anp.sum(targets*anp.log(predictions+1e-9))#/N
    return ce

cross_entropy_grad2 = grad(cross_entropy2,0)

def cross_ent_d(y, yhat):
    return yhat - y

# @njit(cache=False,fastmath=True)  
def Softmax_broadcast(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def Softmax_grad2(x): 
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    s = Softmax(x)
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m


def Softmax_jacob(x):
    s = Softmax(x)
    return np.einsum('ij,jk->ijk', s, np.eye(s.shape[-1])) - np.einsum('ij,ik->ijk', s, s)
#     return contract('ij,jk->ijk', s, np.eye(s.shape[-1])) - contract('ij,ik->ijk', s, s) #SLOWER

@njit(cache=True,fastmath=True)
def Softmax_grad(x): # Best implementation (VERY FAST)
    '''Returns the jacobian of the Softmax function for the given set of inputs.
    Inputs:
    x: should be a 2d array where the rows correspond to the samples
        and the columns correspond to the nodes.
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

# @njit(cache=False,fastmath=True)
def Jsoftmax(X):
    sh = X.shape
    sm = Softmax(X)
    DM = sm.reshape(sh[0],-1,1) * np.diag(np.ones(sh[1])) # Diagonal matrices
#     OP = np.matmul(sm.reshape(sh[0],-1,1), sm.reshape(sh[0],1,-1)) # Outer products
#     OP = np.dot(sm.reshape(sh[0],-1,1), sm.reshape(sh[0],1,-1))
    OP = sm.reshape(sh[0],-1,1) @ sm.reshape(sh[0],1,-1)
    Jsm = DM - OP
    return Jsm


# @njit('float32, (float32[:]),(float32[:]), (float32[:]), (float32[:])',cache=False,fastmath=True, parallel=True)
@njit(cache=True,fastmath=True, parallel=False)
def updateWeightsBiases(eeta, weights, derWeights, biases, derBiases):
    newWeights = weights
    newBiases = biases
    for i in range(nLayers):
#         newWeights[i] = weights[i].astype('float32')-eeta*derWeights[i].astype('float32')
#         newBiases[i] = biases[i].astype('float32')-eeta*derBiases[i].astype('float32')
        newWeights[i] = weights[i]#.astype('float32')-eeta*derWeights[i].astype('float32')
        newBiases[i] = biases[i]#.astype('float32')-eeta*derBiases[i].astype('float32')
    return newWeights, newBiases


# @njit(cache=False,fastmath=True)
def forward_feed(x, nLayers, weights, biases, activationFunc):
    '''
    Performs forward feed 
    Parameters:
        x: matrix (numpy 2D array) of input arrays of size mxn; m: batch size; n: no. of inputs/features
        nLayers: the no. of layers in the network excluding the input layer
        weights: python list of weights; 
            the size of the list is equal to the number of layers (excluding the input layer)
            the weights are numpy 2D ndarrays of size (k,n) where k: is the number of nodes in the current layer 
            and n: is the number of nodes in the preceeding layer ;
            weights (matrix) of the ith layer are accessed as weights[i]
        biases: python list of biases; 
            the size of the list is equal to the number of layers (excluding the input layer)
            the biases are numpy 2D ndarrays of size (k,1) where k: is the number of nodes in the current layer 
            and n: is the number of nodes in the preceeding layer ;
            biases (matrix) of the ith layer are accessed as biases[i]
        activationFunc: a list of the names of suppoted activation functions;
            the size of the list is equal to the number of layers (excluding the input layer)

    Returns:
        a: the list of activated outputs coming out of each layer (including the input layer);
            the size of the list is equal to the number of layers+1 (i.e., the input layer is included)
        z: the list of inputs going into each layer (excluding the input layer);
            the size of the list is equal to the number of layers (excluding the input layer)
        
    '''
    # Forward feed begin
    a = [None] * (nLayers+1)
    z = [None] * nLayers
    a[0] = x
    for l in range(1,nLayers+1):
        z[l-1] = np.einsum('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
#         z[l-1] = contract('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        actFunc_layer = act_func_dict[activationFunc[l-1]] #Activation function for this layer
        a[l] = actFunc_layer(z[l-1])
        
    return a, z

@njit(cache=True,fastmath=True,parallel=False)
def tempEval(a,b,n):
    out = np.empty((n,a.shape[1]),dtype=np.float32)
    for i in range(n):
#     for i in prange(n):
        out[i] = np.dot(a.T, b[i,:]).T
    return out

@njit(cache=False,fastmath=True, parallel=False) #Works faster without tha parallel flag
def softmaxTimesVector(a,b):
    output = np.zeros((a.shape[0],a.shape[1]),dtype=np.float32)
    for i in prange(a.shape[0]):
#         a_temp = 
#         output[i,:] = np.dot(a[i,:,:], b[i,:]).T 
        output[i] = np.dot(a[i], b[i])
    return output

@njit(cache=True,fastmath=True)
def softmaxTimesVector2(a,b):
    output = np.zeros((a.shape[0],a.shape[1]),dtype=np.float32)
    for i in range(a.shape[0]):
#         a_temp = 
        output[i,:] = np.dot(a[i,:,:], b[i,:]).T 
#         output[i] = np.dot(a[i], b[i])
    return output


# @njit(cache=False,fastmath=True)
def back_propagation(z, a, sigmaPrime, nLayers, nSamples, weights, biases, eeta, dc_daL):
    '''
    z: list of input vectors (different sizes) at each layer
    a: list of output vectors after the application of act func (different sizes) at each layer
    sigmaPrime: the function that gives the derivative of the activation function
    dc_daL: a vector that gives the derivative of the Cost function wrt to the output
             vector coming out of the output layer
    '''
    nSamples = a[0].shape[0]
    delta = [None] * (nLayers+1)
    derWeights = [None] * nLayers
    derBiases = [None] * nLayers

    sigmaPrime_layer = act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]).astype(np.float32),dc_daL.astype(np.float32))
    else:
        delta[nLayers] = sigmaPrime_layer(z[nLayers-1])*dc_daL

    newWeights = weights[:]#.copy()
    newBiases = biases[:]#.copy()
    
    derWeights[nLayers-1] = np.einsum('ji,jk->ik',delta[nLayers],a[nLayers-1])
    newWeights[nLayers-1] = weights[nLayers-1] - eeta*derWeights[nLayers-1]
#     derWeights[nLayers-1] = contract('ji,jk->ik',delta[nLayers],a[nLayers-1])
    derBiases[nLayers-1] = np.sum(delta[nLayers],axis=0)
    newBiases[nLayers-1] = biases[nLayers-1] - eeta*derBiases[nLayers-1]
    
    for l in range(nLayers-1,0,-1):
        temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
#         temp = tempEval(np.float32(weights[l]),np.float32(delta[l+1]),nSamples)
#         temp = np.dot(weights[l].T, list(delta[l+1].T)).T # Slower
        sigmaPrime_layer = act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]).astype(np.float32),temp)
        else:    
            delta[l] = sigmaPrime_layer(z[l-1])*temp
#                 
        derWeights[l-1] = np.einsum('ji,jk->ik',delta[l],a[l-1])
#         derWeights[l-1] = contract('ji,jk->ik',delta[l],a[l-1])
        derBiases[l-1] = np.asarray(np.sum(delta[l],axis=0))
        newWeights[l-1] = weights[l-1] - eeta*derWeights[l-1] 
        newBiases[l-1] = biases[l-1] - eeta*derBiases[l-1]
    return derWeights, derBiases, newWeights, newBiases


# @njit(cache=False,fastmath=True)
def back_propagation_fast(z, a, sigmaPrime, nLayers, nSamples, weights, biases, eeta, dc_daL, opt_einsum_expr):
    '''
    z: list of input vectors (different sizes) at each layer
    a: list of output vectors after the application of act func (different sizes) at each layer
    sigmaPrime: the function that gives the derivative of the activation function
    dc_daL: a vector that gives the derivative of the Cost function wrt to the output
             vector coming out of the output layer
    '''
    nSamples = a[0].shape[0]
    delta = [None] * (nLayers+1)
    derWeights = [None] * nLayers
    derBiases = [None] * nLayers
    
    sigmaPrime_layer = act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]).astype(np.float32),dc_daL.astype(np.float32))
#         delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]),dc_daL)
    else:
        delta[nLayers] = sigmaPrime_layer(z[nLayers-1])*dc_daL
    
    newWeights = weights[:]#.copy()
    newBiases = biases[:]#.copy()
    
    derWeights[nLayers-1] = opt_einsum_expr[0](delta[nLayers],a[nLayers-1])
#     derWeights[nLayers-1] = contract('ji,jk->ik',delta[nLayers],a[nLayers-1])
    newWeights[nLayers-1] = weights[nLayers-1] - eeta*derWeights[nLayers-1]
    derBiases[nLayers-1] = np.sum(delta[nLayers],axis=0)
    newBiases[nLayers-1] = biases[nLayers-1] - eeta*derBiases[nLayers-1]
    ioptexpr=1
    for l in range(nLayers-1,0,-1):
        temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
#         temp = tempEval(np.float32(weights[l]),np.float32(delta[l+1]),nSamples)
#         temp = tempEval(weights[l],delta[l+1],nSamples)
#         temp = np.dot(weights[l].T, list(delta[l+1].T)).T # Slower
        sigmaPrime_layer = act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]).astype(np.float32),temp)
#             delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]),temp)
        else:    
            delta[l] = sigmaPrime_layer(z[l-1])*temp
            
        derWeights[l-1] = opt_einsum_expr[ioptexpr](delta[l],a[l-1])
        ioptexpr=ioptexpr+1
#         derWeights[l-1] = contract('ji,jk->ik',delta[l],a[l-1])
        derBiases[l-1] = np.asarray(np.sum(delta[l],axis=0))
        newWeights[l-1] = weights[l-1] - eeta*derWeights[l-1] 
        newBiases[l-1] = biases[l-1] - eeta*derBiases[l-1]
    
    return derWeights, derBiases, newWeights, newBiases

def generateExpressions(nLayers, nSamples, z, a, dc_daL, sigmaPrime, weights):
    delta = [None] * (nLayers+1)
    opt_einsum_expr = []
    
    sigmaPrime_layer = act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]).astype(np.float32),dc_daL.astype(np.float32))
    else:
        delta[nLayers] = sigmaPrime_layer(z[nLayers-1])*dc_daL
        
    opt_einsum_expr.append(contract_expression('ji,jk->ik',delta[nLayers].shape,a[nLayers-1].shape))
    for l in range(nLayers-1,0,-1):
        temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        
        sigmaPrime_layer = act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]).astype(np.float32),temp)
        else:    
            delta[l] = sigmaPrime_layer(z[l-1])*temp
        
        opt_einsum_expr.append(contract_expression('ji,jk->ik',delta[l].shape,a[l-1].shape))
    return opt_einsum_expr

# @njit(cache=False,fastmath=True)
def nn_optimize(nEpochs, batchSize, eeta, inputs, outputs, weights, biases, activationFunc, nLayers, errorFunc, gradErrorFunc,miniterEpoch=1,miniterBatch=100):
    '''
    Performs the optimization of neural network weights and biases using Stochastic gradient descent.
    Parameters:
        nEpochs: no. of epochs (positive integer)
        batchSize: no. of samples in the batch (integer)
        eeta: learning rate (float)
        inputs: batch of inputs (2D numpy ndarray of shape (N,k) where N: no. of samples, and k: no. of input nodes)
        outputs: batch of outputs (2D numpy ndarray of shape (N,k) where N: no. of samples, and k: no. of output nodes)
        weights: python list of weights; 
            the size of the list is equal to the number of layers (excluding the input layer)
            the weights are numpy 2D ndarrays of size (k,n) where k: is the number of nodes in the current layer 
            and n: is the number of nodes in the preceeding layer ;
            weights (matrix) of the ith layer are accessed as weights[i]
        biases: python list of biases; 
            the size of the list is equal to the number of layers (excluding the input layer)
            the biases are numpy 2D ndarrays of size (k,1) where k: is the number of nodes in the current layer 
            and n: is the number of nodes in the preceeding layer ;
            biases (matrix) of the ith layer are accessed as biases[i]
        activationFunc: a list of the names of suppoted activation functions;
            the size of the list is equal to the number of layers (excluding the input layer)
        nLayers: no. of layers in the network excluding the input layer (positive integer)
        errorFunc: the loss function to be used (python function)
        gradErrorFunc: the function that returns the gradient/jacobian of the loss function (python function)
        miniterEpoch: this parameter governs the rate at which the progress bar corresponding to the epochs is updated;
            default value is 1, i.e., the progress bar is updated every epoch.
        miniterBatch: this parameter governs the rate at which the progress bar corresponding to the sample batches is updated;
            default value is 100, i.e., the progress bar is updated every 100 batches.
    Returns:
        weights: the list of optimized weights; the size of the list is equal to the number of layers (excluding the input layer)
        biases: the list of optimized biases the size of the list is equal to the number of layers (excluding the input layer)
        errors: the list of errors at evey epoch
    '''
    errors=[]
    forwardFeedDuration = 0.0
    backPropDuration = 0.0
    updateWeightsDuration = 0.0
    gradErrorDuration = 0.0
    nBatches = int(inputs.shape[0]/batchSize)
    
    for iEpoch in tqdm(range(nEpochs),leave=True,miniters=miniterEpoch):
        errorEpoch = 0.0
        for iBatch in tqdm(range(nBatches),leave=False,miniters=miniterBatch):
            offset = iBatch*batchSize
            x = inputs[offset:offset + batchSize,:]# Input vector
            outExpected = outputs[offset:offset + batchSize,:] # Expected output
#         start = timer()
            # Perform Forward feed and get the outputs at each layers and the inputs at each layer
            a, z = forward_feed(x, nLayers, weights, biases, activationFunc)
#         forwardFeedDuration = forwardFeedDuration + timer() - start
   
            # Error
            errorBatch = errorFunc(a[nLayers],outExpected)
            # Average it over the samples in the batch
            errorEpoch += errorBatch/batchSize
    #         print('Error:\n', errorFunc(a[nLayers],outExpected))
        # Get the derivative of the output cost function wrt to the output vector of the output layer
        # The input arguments should always be an array
#         start = timer()
            dc_daL = gradErrorFunc(a[nLayers], outExpected)
            # Average it out
            dc_daL = dc_daL/batchSize
#         gradErrorDuration = gradErrorDuration + timer() - start
    #         print(dc_daL)
    #         print('Gradient of the error function wrt to the predicted output of the output layer\n',dc_daL)
            # Perform Back Propagation and get the derivatives wrt the weights and biases
#         start = timer()
            derWeights, derBiases, weights, biases = back_propagation(z, a, activationFunc, nLayers, batchSize, weights, biases, eeta, dc_daL)
#         backPropDuration = backPropDuration + timer() - start
  


        # Average over the batches
        errors.append(errorEpoch/nBatches)
        
        if(iEpoch==0):
            print('Average Error with initial weights and biases:', errorEpoch/nBatches)
    
    
    return weights, biases, errors

# @njit(cache=False,fastmath=True)
def nn_optimize_fast(nEpochs, batchSize, eeta, inputs, outputs, weights, biases, activationFunc, nLayers, errorFunc, gradErrorFunc,miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
    '''
    Performs the optimization of neural network weights and biases using Stochastic gradient descent.
    Parameters:
        nEpochs: no. of epochs (positive integer)
        batchSize: no. of samples in the batch (integer)
        eeta: learning rate (float)
        inputs: batch of inputs (2D numpy ndarray of shape (N,k) where N: no. of samples, and k: no. of input nodes)
        outputs: batch of outputs (2D numpy ndarray of shape (N,k) where N: no. of samples, and k: no. of output nodes)
        weights: python list of weights; 
            the size of the list is equal to the number of layers (excluding the input layer)
            the weights are numpy 2D ndarrays of size (k,n) where k: is the number of nodes in the current layer 
            and n: is the number of nodes in the preceeding layer ;
            weights (matrix) of the ith layer are accessed as weights[i]
        biases: python list of biases; 
            the size of the list is equal to the number of layers (excluding the input layer)
            the biases are numpy 2D ndarrays of size (k,1) where k: is the number of nodes in the current layer 
            and n: is the number of nodes in the preceeding layer ;
            biases (matrix) of the ith layer are accessed as biases[i]
        activationFunc: a list of the names of suppoted activation functions;
            the size of the list is equal to the number of layers (excluding the input layer)
        nLayers: no. of layers in the network excluding the input layer (positive integer)
        errorFunc: the loss function to be used (python function)
        gradErrorFunc: the function that returns the gradient/jacobian of the loss function (python function)
        miniterEpoch: this parameter governs the rate at which the progress bar corresponding to the epochs is updated;
            default value is 1, i.e., the progress bar is updated every epoch.
        miniterBatch: this parameter governs the rate at which the progress bar corresponding to the sample batches is updated;
            default value is 100, i.e., the progress bar is updated every 100 batches.
    Returns:
        weights: the list of optimized weights; the size of the list is equal to the number of layers (excluding the input layer)
        biases: the list of optimized biases the size of the list is equal to the number of layers (excluding the input layer)
    '''
    errors=[]
    nBatches = int(inputs.shape[0]/batchSize)
    for iEpoch in tqdm(range(nEpochs),leave=True,miniters=miniterEpoch):
        errorEpoch = 0.0
        # for iBatch in range(nBatches):
        for iBatch in tqdm(range(nBatches),leave=False,miniters=miniterBatch,disable=not(batchProgressBar)):
            offset = iBatch*batchSize
            x = inputs[offset:offset + batchSize,:]# Input vector
          
            outExpected = outputs[offset:offset + batchSize,:] # Expected output
            # Perform Forward feed and get the outputs at each layers and the inputs at each layer
            a, z = forward_feed(x, nLayers, weights, biases, activationFunc)
          
        
            # Error
            errorBatch = errorFunc(a[nLayers],outExpected)
            # Average it over the samples in the batch
            errorEpoch += errorBatch/batchSize
            # Get the derivative of the output cost function wrt to the output vector of the output layer
            # The input arguments should always be an array
            dc_daL = gradErrorFunc(a[nLayers], outExpected)
            # Average it out
            dc_daL = dc_daL/batchSize
            if iEpoch==0 and iBatch==0:
                opt_expr = generateExpressions(nLayers, batchSize, z, a, dc_daL, activationFunc, weights)
   
            # Perform Back Propagation and get the derivatives wrt the weights and biases
            derWeights, derBiases, weights, biases = back_propagation_fast(z, a, activationFunc, nLayers, batchSize, weights, biases, eeta, dc_daL,opt_expr)



        # Average over the batches
        errors.append(errorEpoch/nBatches)
        
        if(iEpoch==0):
            print('Average Error with initial weights and biases:', errorEpoch/nBatches)
    

        
    return weights, biases, errors


act_func_dict = {'Sigmoid':Sigmoid,'ReLU':ReLU,'Softmax':Softmax}
act_func_grad_dict = {'Sigmoid':Sigmoid_grad,'ReLU':ReLU_grad,'Softmax':Softmax_grad}

