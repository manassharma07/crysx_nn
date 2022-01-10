from numba.core import errors
import numpy as np
# print(np.__version__)
# from autograd import numpy as anp
# from autograd import grad, elementwise_grad, jacobian
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
import crysx_nn.utils as utils

try:
    import cupy as cp                     
except ImportError:
    print('CuPy could not be imported!')



def init_params(nInputs, neurons_per_layer, method='random2',dtype='float32'):
    '''
    TODO:Specify float32 or float64 precision
    '''
    # References: 
    # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    # A very good read, for implementing future methods: https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    # TODO Fixup init https://paperswithcode.com/method/fixup-initialization
    nLayers = len(neurons_per_layer)
    weights = [None] * (nLayers)
    biases = [None] * (nLayers)
    for i in range(nLayers):
        if method=='random1':
            # Initialize weights with random numbers [0.0,1.0]
            if i==0:
                weights[i] = np.random.uniform(low=0.0, high=1.0, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.uniform(low=0.0, high=1.0, size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='random2':
            # Initialize weights with random numbers [-0.3,0.3]
            if i==0:
                weights[i] = np.random.uniform(low=-0.3, high=0.3, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.uniform(low=-0.3, high=0.3, size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        if method=='random3':
            # Initialize weights with random numbers [-1.0,1.0]
            if i==0:
                weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.uniform(low=-1.0, high=1.0, size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        if method=='Xavier':
            # Reference: https://paperswithcode.com/method/xavier-initialization
            # Initialize weights with random numbers [-1/sqrt(N),1/sqrt(N)] where N is the number of nodes
            if i==0:
                sqrtN = np.sqrt(nInputs)
                weights[i] = np.random.uniform(low=-1./sqrtN, high=1./sqrtN, size=(neurons_per_layer[i], nInputs))
            else:
                sqrtN = np.sqrt(neurons_per_layer[i-1])
                weights[i] = np.random.uniform(low=-1./sqrtN, high=1./sqrtN, size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='NormXavier':
            # Reference: https://paperswithcode.com/method/xavier-initialization
            # Initialize weights with random numbers [-1/sqrt(N),1/sqrt(N)] where N is the number of nodes
            if i==0:
                sqrtN = np.sqrt(nInputs)
                sqrtM = np.sqrt(neurons_per_layer[i])
                weights[i] = np.random.uniform(low=-6./(sqrtN+sqrtM), high=6./(sqrtN+sqrtM), size=(neurons_per_layer[i], nInputs))
            else:
                sqrtN = np.sqrt(neurons_per_layer[i-1])
                sqrtM = np.sqrt(neurons_per_layer[i])
                weights[i] = np.random.uniform(low=-6./(sqrtN+sqrtM), high=6./(sqrtN+sqrtM), size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='He':
            # Reference: https://paperswithcode.com/method/xavier-initialization
            # Initialize weights with random numbers [-1/sqrt(N),1/sqrt(N)] where N is the number of nodes
            if i==0:
                weights[i] = np.random.normal(loc=0.0, scale=np.sqrt(2./nInputs), size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = np.random.normal(loc=0.0, scale=np.sqrt(2./neurons_per_layer[i-1]), size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        # Initialize biases
        biases[i] = np.zeros(neurons_per_layer[i])

    # Ensure the dtype is same as that expected
    if dtype=='float32':
        for i in range(len(weights)):
            weights[i] = weights[i].astype(np.float32)
            biases[i] = biases[i].astype(np.float32)

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
        z[l-1] = contract('ij,kj->ik',a[l-1],weights[l-1],dtype=x.dtype)+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        actFunc_layer = utils.act_func_dict[activationFunc[l-1]] #Activation function for this layer
        a[l] = actFunc_layer(z[l-1]).astype(x.dtype)
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
    output = contract('ijk,ik->ij',a,b, dtype=a.dtype)
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

    sigmaPrime_layer = utils.act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
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
        sigmaPrime_layer = utils.act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
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
    
    sigmaPrime_layer = utils.act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]),dc_daL).astype(z[0].dtype)
#         delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]),dc_daL)
    else:
        delta[nLayers] = (sigmaPrime_layer(z[nLayers-1])*dc_daL).astype(z[0].dtype)
    
    newWeights = weights[:]#.copy()
    newBiases = biases[:]#.copy()
    
    derWeights[nLayers-1] = opt_einsum_expr[0](delta[nLayers],a[nLayers-1])
#     derWeights[nLayers-1] = contract('ji,jk->ik',delta[nLayers],a[nLayers-1])
    newWeights[nLayers-1] = weights[nLayers-1] - eeta*derWeights[nLayers-1]
    derBiases[nLayers-1] = np.sum(delta[nLayers],axis=0)
    newBiases[nLayers-1] = biases[nLayers-1] - eeta*derBiases[nLayers-1]
    ioptexpr=1
    for l in range(nLayers-1,0,-1):
        temp = contract('ik,lk->li',weights[l].T, delta[l+1], dtype=z[0].dtype)
        # temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        # temp = tempEval(np.float32(weights[l]),np.float32(delta[l+1]),nSamples)
#         temp = tempEval(weights[l],delta[l+1],nSamples)
#         temp = np.dot(weights[l].T, list(delta[l+1].T)).T # Slower
        sigmaPrime_layer = utils.act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]),temp).astype(z[0].dtype)
#             delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]),temp)
        else:    
            delta[l] = (sigmaPrime_layer(z[l-1])*temp).astype(z[0].dtype)
            
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
    
    sigmaPrime_layer = utils.act_func_grad_dict[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]),dc_daL).astype(z[0].dtype)
    else:
        delta[nLayers] = (sigmaPrime_layer(z[nLayers-1])*dc_daL).astype(z[0].dtype)
        
    opt_einsum_expr.append(contract_expression('ji,jk->ik',delta[nLayers].shape,a[nLayers-1].shape,dtype=z[0].dtype))
    for l in range(nLayers-1,0,-1):
        temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        
        sigmaPrime_layer = utils.act_func_grad_dict[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]),temp).astype(z[0].dtype)
        else:    
            delta[l] = (sigmaPrime_layer(z[l-1])*temp).astype(z[0].dtype)
        
        opt_einsum_expr.append(contract_expression('ji,jk->ik',delta[l].shape,a[l-1].shape,dtype=z[0].dtype))
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
        weights =[]#, dummy = init_params()
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


# act_func_dict = {'Sigmoid':activation.Sigmoid,'ReLU':activation.ReLU,'Softmax':activation.Softmax}
# act_func_grad_dict = {'Sigmoid':activation.Sigmoid_grad,'ReLU':activation.ReLU_grad,'Softmax':activation.Softmax_grad}

##------------------CUPY----------------------
# act_func_dict_cupy = {'Sigmoid':activation.Sigmoid_cupy,'ReLU':activation.ReLU_cupy,'Softmax':activation.Softmax_cupy}
# act_func_grad_dict_cupy = {'Sigmoid':activation.Sigmoid_grad_cupy,'ReLU':activation.ReLU_grad_cupy,'Softmax':activation.Softmax_grad_cupy}

def init_params_cupy(nInputs, neurons_per_layer, method='random2', dtype='float32'):
    # References: 
    # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    # A very good read, for implementing future methods: https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    # TODO Fixup init https://paperswithcode.com/method/fixup-initialization
    nLayers = len(neurons_per_layer)
    weights = [None] * (nLayers)
    biases = [None] * (nLayers)
    for i in range(nLayers):
        if method=='random1':
            # Initialize weights with random numbers [0.0,1.0]
            if i==0:
                weights[i] = cp.random.uniform(low=0.0, high=1.0, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = cp.random.uniform(low=0.0, high=1.0, size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='random2':
            # Initialize weights with random numbers [-0.3,0.3]
            if i==0:
                weights[i] = cp.random.uniform(low=-0.3, high=0.3, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = cp.random.uniform(low=-0.3, high=0.3, size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        if method=='random3':
            # Initialize weights with random numbers [-1.0,1.0]
            if i==0:
                weights[i] = cp.random.uniform(low=-1.0, high=1.0, size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = cp.random.uniform(low=-1.0, high=1.0, size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        if method=='Xavier':
            # Reference: https://paperswithcode.com/method/xavier-initialization
            # Initialize weights with random numbers [-1/sqrt(N),1/sqrt(N)] where N is the number of nodes
            if i==0:
                sqrtN = cp.sqrt(nInputs)
                weights[i] = cp.random.uniform(low=-1./sqrtN, high=1./sqrtN, size=(neurons_per_layer[i], nInputs))
            else:
                sqrtN = cp.sqrt(neurons_per_layer[i-1])
                weights[i] = cp.random.uniform(low=-1./sqrtN, high=1./sqrtN, size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='NormXavier':
            # Reference: https://paperswithcode.com/method/xavier-initialization
            # Initialize weights with random numbers [-1/sqrt(N),1/sqrt(N)] where N is the number of nodes
            if i==0:
                sqrtN = cp.sqrt(nInputs)
                sqrtM = cp.sqrt(neurons_per_layer[i])
                weights[i] = cp.random.uniform(low=-6./(sqrtN+sqrtM), high=6./(sqrtN+sqrtM), size=(neurons_per_layer[i], nInputs))
            else:
                sqrtN = cp.sqrt(neurons_per_layer[i-1])
                sqrtM = cp.sqrt(neurons_per_layer[i])
                weights[i] = cp.random.uniform(low=-6./(sqrtN+sqrtM), high=6./(sqrtN+sqrtM), size=(neurons_per_layer[i], neurons_per_layer[i-1]))

        if method=='He':
            # Reference: https://paperswithcode.com/method/xavier-initialization
            # Initialize weights with random numbers [-1/sqrt(N),1/sqrt(N)] where N is the number of nodes
            if i==0:
                weights[i] = cp.random.normal(loc=0.0, scale=cp.sqrt(2./nInputs), size=(neurons_per_layer[i], nInputs))
            else:
                weights[i] = cp.random.normal(loc=0.0, scale=cp.sqrt(2./neurons_per_layer[i-1]), size=(neurons_per_layer[i], neurons_per_layer[i-1]))
        
        # Initialize biases
        biases[i] = cp.zeros(neurons_per_layer[i])

    # Ensure the dtype is same as that expected
    if dtype=='float32':
        for i in range(len(weights)):
            weights[i] = weights[i].astype(cp.float32)
            biases[i] = biases[i].astype(cp.float32)

    return weights, biases


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
        # z[l-1] = cp.einsum('ij,kj->ik',a[l-1],weights[l-1])+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        z[l-1] = contract('ij,kj->ik',a[l-1],weights[l-1],backend='cupy')+biases[l-1] #np.dot(a[l-1],weights[l-1])#np.asarray(biases[l-1] + np.dot(a[l-1],weights[l-1])) #np.einsum('jk,k->j',weights[l-1],a[l-1])s #weights[l-1]*a[l-1]
        actFunc_layer = utils.act_func_dict_cupy[activationFunc[l-1]] #Activation function for this layer
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

    sigmaPrime_layer = utils.act_func_grad_dict_cupy[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector_cupy(sigmaPrime_layer(z[nLayers-1]).astype(cp.float32),dc_daL.astype(cp.float32))
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
        sigmaPrime_layer = utils.act_func_grad_dict_cupy[sigmaPrime[l-1]] # Act func gradient for this layer
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
    # output = cp.einsum('ijk,ik->ij',a,b)
    output = contract('ijk,ik->ij',a,b, backend='cupy')
    # output = (a @ b[..., cp.newaxis])[..., 0]
    return output

def back_propagation_fast_cupy(z, a, sigmaPrime, nLayers, nSamples, weights, biases, eeta, dc_daL, opt_einsum_expr):
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
    
    sigmaPrime_layer = utils.act_func_grad_dict_cupy[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector_cupy(sigmaPrime_layer(z[nLayers-1]).astype(z[0].dtype),dc_daL.astype(z[0].dtype))
#         delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]),dc_daL)
    else:
        delta[nLayers] = sigmaPrime_layer(z[nLayers-1])*dc_daL
    
    newWeights = weights[:]#.copy()
    newBiases = biases[:]#.copy()
    
    derWeights[nLayers-1] = opt_einsum_expr[0](delta[nLayers],a[nLayers-1], backend='cupy')
#     derWeights[nLayers-1] = contract('ji,jk->ik',delta[nLayers],a[nLayers-1])
    newWeights[nLayers-1] = weights[nLayers-1] - eeta*derWeights[nLayers-1]
    derBiases[nLayers-1] = cp.sum(delta[nLayers],axis=0)
    newBiases[nLayers-1] = biases[nLayers-1] - eeta*derBiases[nLayers-1]
    ioptexpr=1
    for l in range(nLayers-1,0,-1):
        temp = contract('ik,lk->li',weights[l].T, delta[l+1], backend='cupy',dtype=z[0].dtype)
        # temp = np.array([np.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        # temp = tempEval(np.float32(weights[l]),np.float32(delta[l+1]),nSamples)
#         temp = tempEval(weights[l],delta[l+1],nSamples)
#         temp = np.dot(weights[l].T, list(delta[l+1].T)).T # Slower
        sigmaPrime_layer = utils.act_func_grad_dict_cupy[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector_cupy(sigmaPrime_layer(z[l-1]).astype(z[0].dtype),temp)
#             delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]),temp)
        else:    
            delta[l] = sigmaPrime_layer(z[l-1])*temp
        derWeights[l-1] = opt_einsum_expr[ioptexpr](delta[l],a[l-1], backend='cupy')
        ioptexpr=ioptexpr+1
#         derWeights[l-1] = contract('ji,jk->ik',delta[l],a[l-1])
        derBiases[l-1] = cp.asarray(cp.sum(delta[l],axis=0))
        newWeights[l-1] = weights[l-1] - eeta*derWeights[l-1] 
        newBiases[l-1] = biases[l-1] - eeta*derBiases[l-1]
    
    return derWeights, derBiases, newWeights, newBiases

def generateExpressions_cupy(nLayers, nSamples, z, a, dc_daL, sigmaPrime, weights):
    delta = [None] * (nLayers+1)
    opt_einsum_expr = []
    
    sigmaPrime_layer = utils.act_func_grad_dict_cupy[sigmaPrime[nLayers-1]] # Act func gradient for this layer
    if sigmaPrime[nLayers-1] =='Softmax':
        delta[nLayers] = softmaxTimesVector(sigmaPrime_layer(z[nLayers-1]).astype(cp.float32),dc_daL.astype(cp.float32))
    else:
        delta[nLayers] = sigmaPrime_layer(z[nLayers-1])*dc_daL
        
    opt_einsum_expr.append(contract_expression('ji,jk->ik',delta[nLayers].shape,a[nLayers-1].shape))
    for l in range(nLayers-1,0,-1):
        temp = cp.array([cp.dot(weights[l].T, delta[l+1][i,:]).T for i in range(nSamples)])
        
        sigmaPrime_layer = utils.act_func_grad_dict_cupy[sigmaPrime[l-1]] # Act func gradient for this layer
        if sigmaPrime[l-1] =='Softmax':
            delta[l] = softmaxTimesVector(sigmaPrime_layer(z[l-1]).astype(cp.float32),temp)
        else:    
            delta[l] = sigmaPrime_layer(z[l-1])*temp
        
        opt_einsum_expr.append(contract_expression('ji,jk->ik',delta[l].shape,a[l-1].shape))
    return opt_einsum_expr

def nn_optimize_cupy(inputs, outputs, activationFunc, nLayers, nEpochs=10, batchSize=None, eeta=0.5, weights=None, biases=None, errorFunc=loss.MSE_loss_cupy, gradErrorFunc=loss.MSE_loss_grad_cupy,miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
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

def nn_optimize_fast_cupy(inputs, outputs, activationFunc, nLayers, nEpochs=10, batchSize=None, eeta=0.5, weights=None, biases=None, errorFunc=loss.MSE_loss_cupy, gradErrorFunc=loss.MSE_loss_grad_cupy,miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
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
            a, z = forward_feed_cupy(x, nLayers, weights, biases, activationFunc)
          
        
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
                opt_expr = generateExpressions_cupy(nLayers, batchSize, z, a, dc_daL, activationFunc, weights)
   
            # Perform Back Propagation and get the derivatives wrt the weights and biases
            derWeights, derBiases, weights, biases = back_propagation_fast_cupy(z, a, activationFunc, nLayers, batchSize, weights, biases, eeta, dc_daL,opt_expr)
            


        # Average over the batches
        errors.append(errorEpoch/nBatches)
        
        if(iEpoch==0):
            print('Average Error with initial weights and biases:', errorEpoch/nBatches)
    

        
    return weights, biases, errors

class nn_model:
    def __init__(self, nInputs=None, neurons_per_layer=None, activation_func_names=None, batch_size=None, device='CPU', init_method='Xavier'): 
        if nInputs is None:
            print('ERROR: You need to specify the number of input nodes.')
            return
        else: 
            self.nInputs = nInputs
        if neurons_per_layer is None:
            print('ERROR: You need to specify the number of neurons per layer (excluding the input layer) and supply it as a list.')
            return
        else: 
            self.neurons_per_layer = neurons_per_layer
        if activation_func_names is None:
            print('ERROR: You need to specify the activation function for each layer and supply it as a list.')
            return
        else: 
            self.activation_func_names = activation_func_names
        print('Note: The model will use the following device for all the computations: ', device)
        
        self.batch_size = batch_size
        self.device = device
        self.init_method = init_method
        if self.device=='CPU':
            self.init_weights, self.init_biases = init_params(self.nInputs, self.neurons_per_layer, method=self.init_method)
        if self.device=='GPU':
            self.init_weights, self.init_biases = init_params_cupy(self.nInputs, self.neurons_per_layer, method=self.init_method)
        self.nLayers = len(neurons_per_layer)
        self.weights = self.init_weights
        self.biases = self.init_biases
        self.errors = []
        self.opt_method = 'SGD'
        self.lr = 0.5

    def init_params(self, method=None):
        if method is None:
            method = self.init_method
        if self.device=='CPU':
            self.init_weights, self.init_biases = init_params(self.nInputs, self.neurons_per_layer, method=method)
        if self.device=='GPU':
            self.init_weights, self.init_biases = init_params_cupy(self.nInputs, self.neurons_per_layer, method=method)
        self.weights = self.init_weights
        self.biases = self.init_biases


    def visualize(self):
        visualize(self.nInputs, self.neurons_per_layer, self.activation_func_names)
        
    def details(self):
        print('----------------------------------------------------------------------------------')
        print('****Neural Network Model Details****')
        print('----------------------------------------------------------------------------------')
        print('Number of input nodes: ', self.nInputs)
        print('Number of layers (hidden+output): ', self.nLayers)
        print('Number of nodes in each layer (hidden & output): ', self.neurons_per_layer)
        print('Activation function for each layer (hidden & output):  ', self.activation_func_names)
        print('Method used for weights and biases initialization:  ', self.init_method)
        print('Batch Size: ', self.batch_size)
        print('Device: ', self.device)
        print('Optimization method: ', self.opt_method)
        print('Learning rate: ', self.lr)
        print('----------------------------------------------------------------------------------')
        
    def optimize(self, inputs, outputs, method=None, lr=None, nEpochs=100,loss_func_name=None, miniterEpoch=1,batchProgressBar=False,miniterBatch=100):
        if method is None:
            method = self.opt_method
        if lr is None:
            lr = self.lr
        if self.device=='CPU':
            if loss_func_name is None:
                loss_func = loss.MSE_loss
                loss_func_grad = loss.MSE_loss_grad
            else:
                loss_func = utils.loss_func_dict[loss_func_name]
                loss_func_grad = utils.loss_func_grad_dict[loss_func_name]
            self.weights, self.biases, self.errors = nn_optimize_fast(inputs, outputs, self.activation_func_names, self.nLayers, nEpochs=nEpochs, batchSize=self.batch_size, eeta=lr, weights=self.weights, biases=self.biases, errorFunc=loss_func, gradErrorFunc=loss_func_grad,miniterEpoch=miniterEpoch,batchProgressBar=batchProgressBar,miniterBatch=miniterBatch)
        if self.device=='GPU':
            if loss_func_name is None:
                loss_func = loss.MSE_loss_cupy
                loss_func_grad = loss.MSE_loss_grad_cupy
            else:
                loss_func = utils.loss_func_dict_cupy[loss_func_name]
                loss_func_grad = utils.loss_func_grad_dict_cupy[loss_func_name]
            self.weights, self.biases, self.errors = nn_optimize_fast_cupy(inputs, outputs, self.activation_func_names, self.nLayers, nEpochs=nEpochs, batchSize=self.batch_size, eeta=lr, weights=self.weights, biases=self.biases, errorFunc=loss_func, gradErrorFunc=loss_func_grad,miniterEpoch=miniterEpoch,batchProgressBar=batchProgressBar,miniterBatch=miniterBatch)
    
    def predict(self, inputs, outputs=None, loss_func_name=None):
        error = 0.0
        nBatches = np.maximum(int(inputs.shape[0]/self.batch_size),1)
        if self.device=='CPU':
            if loss_func_name is None:
                loss_func = loss.MSE_loss
                loss_func_grad = loss.MSE_loss_grad
            else:
                loss_func = utils.loss_func_dict[loss_func_name]
                loss_func_grad = utils.loss_func_grad_dict[loss_func_name]
            for iBatch in range(nBatches):
                offset = iBatch*self.batch_size
                x = inputs[offset:offset + self.batch_size,:]# Input vector
                
                # Forward feed with optimized weights
                # Perform Forward feed and get the outputs at each layers and the inputs at each layer
                a, z = forward_feed(x, self.nLayers, self.weights, self.biases, self.activation_func_names)
                new_outputs = a[self.nLayers] 
                if outputs is not None:
                    outExpected = outputs[offset:offset + self.batch_size,:] # Expected output
                    # New Error
                    error += loss_func(new_outputs, outExpected)/self.batch_size
        if self.device=='GPU':
            if loss_func_name is None:
                loss_func = loss.MSE_loss_cupy
                loss_func_grad = loss.MSE_loss_grad_cupy
            else:
                loss_func = utils.loss_func_dict_cupy[loss_func_name]
                loss_func_grad = utils.loss_func_grad_dict_cupy[loss_func_name]
            for iBatch in range(nBatches):
                offset = iBatch*self.batch_size
                x = inputs[offset:offset + self.batch_size,:]# Input vector
                
                # Forward feed with optimized weights
                # Perform Forward feed and get the outputs at each layers and the inputs at each layer
                a, z = forward_feed_cupy(x, self.nLayers, self.weights, self.biases, self.activation_func_names)
                new_outputs = a[self.nLayers] 
                if outputs is not None:
                    outExpected = outputs[offset:offset + self.batch_size,:] # Expected output
                    # New Error
                    # New Error
                    error += loss_func(new_outputs, outExpected)/self.batch_size
        if outputs is None:
            return new_outputs
        else:
            return new_outputs, error/nBatches