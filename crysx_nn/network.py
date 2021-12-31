import numpy as np
# print(np.__version__)
from autograd import numpy as anp
from autograd import grad, elementwise_grad, jacobian
# if hasattr(__builtins__,'__IPYTHON__'):
#     from tqdm.notebook import tqdm
#     print('Using notebook tqdm')
# else:
#     from tqdm import tqdm
#     print('Using simple tqdm')

from tqdm.autonotebook import tqdm
# from tqdm.auto import tqdm
import numexpr as ne
from opt_einsum import contract, contract_expression
from numba import vectorize,jit,njit,prange,set_num_threads,get_num_threads 
from nnv import NNV
import matplotlib.pyplot as plt
from operator import add
import crysx_nn.loss as loss
import crysx_nn.activation as activation

try:
    import cupy as cp                     
except ImportError:
    print('CuPy could not be imported!')



def initParams(nInputs, neurons_per_layer, activation_func_names):
    nLayers = len(neurons_per_layer)
    weights = []
    biases = []

    return weights, biases


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




# @njit('float32, (float32[:]),(float32[:]), (float32[:]), (float32[:])',cache=False,fastmath=True, parallel=True)
@njit(cache=True,fastmath=True, parallel=False)
def updateWeightsBiases(nLayers, eeta, weights, derWeights, biases, derBiases):
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
        # z[l-1] = np.einsum('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        z[l-1] = contract('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        actFunc_layer = act_func_dict[activationFunc[l-1]] #Activation function for this layer
        a[l] = actFunc_layer(z[l-1])
        
    return a, z

@njit(cache=True,fastmath=True,parallel=True) #Works faster without tha parallel flag
def tempEval(a,b,n):
    out = np.empty((n,a.shape[1]),dtype=np.float32)
    for i in prange(n):
#     for i in prange(n):
        out[i] = np.dot(a.T, b[i,:]).T
    return out

@njit(cache=True,fastmath=True, parallel=True) #Works faster without tha parallel flag
def softmaxTimesVector_(a,b):
    output = np.zeros((a.shape[0],a.shape[1]),dtype=np.float32)
    for i in prange(a.shape[0]):
        output[i] = np.dot(a[i], b[i])
    return output


def softmaxTimesVector(a,b): 
    # Reference: https://stackoverflow.com/questions/59289754/numpy-multiply-3d-array-with-2d-array
    ## Both the following methods are equally fast and give correct results
    output = contract('ijk,ik->ij',a,b)
    # output = (a @ b[..., np.newaxis])[..., 0]
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
        # print(weights[l].T.shape)
        # print(delta[l+1].shape)
        # return
        temp = np.einsum('ik,lk->li',weights[l].T, delta[l+1])
        # temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        # print(temp.shape)
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
        temp = contract('ik,lk->li',weights[l].T, delta[l+1])
        # temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        # temp = tempEval(np.float32(weights[l]),np.float32(delta[l+1]),nSamples)
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
def nn_optimize(inputs, outputs, activationFunc, nLayers, nEpochs=10, batchSize=None, eeta=0.5, weights=None, biases=None, errorFunc=loss.MSE_loss, gradErrorFunc=loss.MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
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
    if batchSize==None:
        batchSize = min(32, inputs.shape[0])
    if weights == None:
        weights = []
    if biases == None:
        biases = []
    errors=[]
    forwardFeedDuration = 0.0
    backPropDuration = 0.0
    updateWeightsDuration = 0.0
    gradErrorDuration = 0.0
    nBatches = int(inputs.shape[0]/batchSize)
    
    for iEpoch in tqdm(range(nEpochs),leave=True,miniters=miniterEpoch):
        errorEpoch = 0.0
        for iBatch in tqdm(range(nBatches),leave=False,miniters=miniterBatch,disable=not(batchProgressBar)):
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
def nn_optimize_fast(inputs, outputs, activationFunc, nLayers, nEpochs=10, batchSize=None, eeta=0.5, weights=None, biases=None, errorFunc=loss.MSE_loss, gradErrorFunc=loss.MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
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
    if batchSize==None:
        batchSize = min(32, inputs.shape[0])
    if weights == None:
        weights = []
    if biases == None:
        biases = []
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


act_func_dict = {'Sigmoid':activation.Sigmoid,'ReLU':activation.ReLU,'Softmax':activation.Softmax}
act_func_grad_dict = {'Sigmoid':activation.Sigmoid_grad,'ReLU':activation.ReLU_grad,'Softmax':activation.Softmax_grad}

##------------------CUPY----------------------
act_func_dict_cupy = {'Sigmoid':activation.Sigmoid_cupy,'ReLU':activation.ReLU_cupy,'Softmax':activation.Softmax_cupy}
act_func_grad_dict_cupy = {'Sigmoid':activation.Sigmoid_grad_cupy,'ReLU':activation.ReLU_grad_cupy,'Softmax':activation.Softmax_grad_cupy}


def forward_feed_cupy(x, nLayers, weights, biases, activationFunc):
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
        z[l-1] = cp.einsum('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
#         z[l-1] = contract('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        actFunc_layer = act_func_dict_cupy[activationFunc[l-1]] #Activation function for this layer
        a[l] = actFunc_layer(z[l-1])
        
    return a, z

def back_propagation_cupy(z, a, sigmaPrime, nLayers, nSamples, weights, biases, eeta, dc_daL):
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

    sigmaPrime_layer = act_func_grad_dict_cupy[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector_cupy(sigmaPrime_layer(z[nLayers-1]).astype(np.float32),dc_daL.astype(np.float32))
    else:
        delta[nLayers] = sigmaPrime_layer(z[nLayers-1])*dc_daL

    newWeights = weights[:]#.copy()
    newBiases = biases[:]#.copy()
    
    derWeights[nLayers-1] = cp.einsum('ji,jk->ik',delta[nLayers],a[nLayers-1])
    newWeights[nLayers-1] = weights[nLayers-1] - eeta*derWeights[nLayers-1]
#     derWeights[nLayers-1] = contract('ji,jk->ik',delta[nLayers],a[nLayers-1])
    derBiases[nLayers-1] = cp.sum(delta[nLayers],axis=0)
    newBiases[nLayers-1] = biases[nLayers-1] - eeta*derBiases[nLayers-1]
    
    for l in range(nLayers-1,0,-1):
        temp = cp.einsum('ik,lk->li',weights[l].T, delta[l+1])
        # temp = cp.array([cp.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
#         temp = tempEval(np.float32(weights[l]),np.float32(delta[l+1]),nSamples)
#         temp = np.dot(weights[l].T, list(delta[l+1].T)).T # Slower
        sigmaPrime_layer = act_func_grad_dict_cupy[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector_cupy(sigmaPrime_layer(z[l-1]).astype(cp.float32),temp)
        else:    
            delta[l] = sigmaPrime_layer(z[l-1])*temp
#                 
        derWeights[l-1] = cp.einsum('ji,jk->ik',delta[l],a[l-1])
#         derWeights[l-1] = contract('ji,jk->ik',delta[l],a[l-1])
        derBiases[l-1] = cp.asarray(np.sum(delta[l],axis=0))
        newWeights[l-1] = weights[l-1] - eeta*derWeights[l-1] 
        newBiases[l-1] = biases[l-1] - eeta*derBiases[l-1]
    return derWeights, derBiases, newWeights, newBiases

def softmaxTimesVector_cupy(a,b):
    # Reference: https://stackoverflow.com/questions/59289754/numpy-multiply-3d-array-with-2d-array
    ## Both the following methods are equally fast and give correct results
    output = cp.einsum('ijk,ik->ij',a,b)
    # output = (a @ b[..., cp.newaxis])[..., 0]
    return output

def nn_optimize_cupy(inputs, outputs, activationFunc, nLayers, nEpochs=10, batchSize=None, eeta=0.5, weights=None, biases=None, errorFunc=loss.MSE_loss, gradErrorFunc=loss.MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
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
    if batchSize==None:
        batchSize = min(32, inputs.shape[0])
    if weights == None:
        weights = []
    if biases == None:
        biases = []
    errors=[]
    forwardFeedDuration = 0.0
    backPropDuration = 0.0
    updateWeightsDuration = 0.0
    gradErrorDuration = 0.0
    nBatches = int(inputs.shape[0]/batchSize)
    
    for iEpoch in tqdm(range(nEpochs),leave=True,miniters=miniterEpoch):
        errorEpoch = 0.0
        for iBatch in tqdm(range(nBatches),leave=False,miniters=miniterBatch,disable=not(batchProgressBar)):
            offset = iBatch*batchSize
            x = inputs[offset:offset + batchSize,:]# Input vector
            outExpected = outputs[offset:offset + batchSize,:] # Expected output
#         start = timer()
            # Perform Forward feed and get the outputs at each layers and the inputs at each layer
            a, z = forward_feed_cupy(x, nLayers, weights, biases, activationFunc)
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
            derWeights, derBiases, weights, biases = back_propagation_cupy(z, a, activationFunc, nLayers, batchSize, weights, biases, eeta, dc_daL)
#         backPropDuration = backPropDuration + timer() - start
  


        # Average over the batches
        errors.append(errorEpoch/nBatches)
        
        if(iEpoch==0):
            print('Average Error with initial weights and biases:', errorEpoch/nBatches)
    
    
    return weights, biases, errors