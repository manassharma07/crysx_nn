#!/usr/bin/env python
# coding: utf-8

# # Simulating logic gates using `crysx_nn`

# In[50]:


import numpy as np
import matplotlib.pyplot as plt


# Import CrysX_nn related stuff

# Import loss functions
from crysx_nn.loss import MSE_loss, MSE_loss_grad
from crysx_nn.loss import BCE_loss, BCE_loss_grad
# Import activation functions
from crysx_nn.activation import Sigmoid, Sigmoid_grad
from crysx_nn.activation import Softmax, Softmax_grad
from crysx_nn.activation import Tanh_offset, Tanh_offset
# Import neural network functions
from crysx_nn.network import back_propagation, back_propagation_fast
from crysx_nn.network import nn_optimize, nn_optimize_fast
from crysx_nn.network import forward_feed

from crysx_nn.network import visualize


# ## Define some network parameters that we will be using later on

# In[51]:


nInputs = 2 # No. of nodes in the input layer
neurons_per_layer = [10,1] # Neurons per layer (excluding the input layer)
activation_func_names = ['Sigmoid', 'Sigmoid']
nLayers = len(neurons_per_layer)
eeta = 0.5
nEpochs=10**4
batchSize = 4 # No. of input samples to process at a time for optimization


# ## Visualize the network

# In[52]:


visualize(nInputs, neurons_per_layer, activation_func_names)


# ## Set the inputs (rows correspond to different set of inputs and columns represent the inputs)

# In[53]:


inputs = np.array([[0.,0.,1.,1.],[0.,1.,0.,1.]]).T.astype('float32')
print(inputs)
print(inputs.dtype)


# ## Outputs for different logic gates

# In[54]:


# AND function
outputAND = np.array([0.,0.,0.,1.])
outputAND = np.asarray([outputAND]).T
# OR function
outputOR = np.array([0.,1.,1.,1.])
outputOR = np.asarray([outputOR]).T
# NAND function
outputNAND = np.array([1.,1.,1.,0.])
outputNAND = np.asarray([outputNAND]).T
# XOR function
outputXOR = np.array([0.,1.,1.,0.])
outputXOR = np.asarray([outputXOR]).T


# In[55]:


print('AND outputs\n', outputAND)
print('OR outputs\n', outputOR)
print('XOR outputs\n', outputXOR)
print('NAND outputs\n', outputNAND)


# ## Define/ generate initial guess and weights

# In[56]:


# Initial guesses for weights
w1 = 0.30
w2 = 0.55
w3 = 0.20
w4 = 0.45
w5 = 0.50
w6 = 0.35
w7 = 0.15
w8 = 0.40
w9 = 0.25

# Initial guesses for biases
b1 = 0.60
b2 = 0.05

# need to use a list instead of a numpy array, since the 
#weight matrices at each layer are not of the same dimensions
weights = [] 
# Weights for layer 1 --> 2
weights.append(np.array([[w1,w4],[w2, w5], [w3, w6]]))
# Weights for layer 2 --> 3
weights.append(np.array([[w7, w8, w9]]))
# List of biases at each layer
biases = []
biases.append(np.array([b1,b1,b1]))
biases.append(np.array([b2]))

weightsOriginal = weights
biasesOriginal = biases

print('Weights matrices: ',weights)
print('Biases: ',biases)


# ## Train neural network for AND gate
# We will use the Mean Squared Error loss function (MSE_loss)

# In[57]:



# Run optimization
optWeights, optBiases, errorPlot = nn_optimize_fast(inputs, outputAND, activation_func_names, nLayers, nEpochs=nEpochs, batchSize=batchSize, eeta=eeta, weights=weightsOriginal, biases=biasesOriginal, errorFunc=MSE_loss, gradErrorFunc=MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100)


# ### Visualize the training loss

# In[58]:


# Plot the error vs epochs
plt.plot(errorPlot)
plt.yscale('log')
plt.show()


# ### Optimized weights and biases

# In[59]:


# Optimized weights and biases
print('Optimized Weights: ', optWeights)
print('Optimized Biases: ', optBiases)


# ### Make predictions using the optimized weights

# In[60]:


# Forward feed with optimized weights
# Perform Forward feed and get the outputs at each layers and the inputs at each layer
a, z = forward_feed(inputs, nLayers, optWeights, optBiases, activation_func_names)
print('Outputs at each layer (including input layer):\n ',a)
print('Inputs at each layer :\n ', z)
# New Error
print('New Average Error with optimized weights:\n', MSE_loss(a[nLayers],outputAND)/batchSize)
print('Input: ',inputs)
print('Output: ',a[nLayers])
print('Expected Output: ',outputAND)


# ## Train neural network for NAND gate
# We will use the Mean Squared Error loss function (MSE_loss)

# In[61]:



# Run optimization
optWeights, optBiases, errorPlot = nn_optimize_fast(inputs, outputNAND, activation_func_names, nLayers, nEpochs=nEpochs, batchSize=batchSize, eeta=eeta, weights=weightsOriginal, biases=biasesOriginal, errorFunc=MSE_loss, gradErrorFunc=MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100)


# ### Visualize the training loss

# In[62]:


# Plot the error vs epochs
plt.plot(errorPlot)
plt.yscale('log')
plt.show()


# ### Optimized weights and biases

# In[63]:


# Optimized weights and biases
print('Optimized Weights: ', optWeights)
print('Optimized Biases: ', optBiases)


# ### Make predictions using the optimized weights

# In[64]:


# Forward feed with optimized weights
# Perform Forward feed and get the outputs at each layers and the inputs at each layer
a, z = forward_feed(inputs, nLayers, optWeights, optBiases, activation_func_names)
print('Outputs at each layer (including input layer):\n ',a)
print('Inputs at each layer :\n ', z)
# New Error
print('New Average Error with optimized weights:\n', MSE_loss(a[nLayers],outputNAND)/batchSize)
print('Input: ',inputs)
print('Output: ',a[nLayers])
print('Expected Output: ',outputNAND)


# ## Train neural network for OR gate
# We will use the Mean Squared Error loss function (MSE_loss)

# In[65]:



# Run optimization
optWeights, optBiases, errorPlot = nn_optimize_fast(inputs, outputOR, activation_func_names, nLayers, nEpochs=nEpochs, batchSize=batchSize, eeta=eeta, weights=weightsOriginal, biases=biasesOriginal, errorFunc=MSE_loss, gradErrorFunc=MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100)


# ### Visualize the training loss

# In[66]:


# Plot the error vs epochs
plt.plot(errorPlot)
plt.yscale('log')
plt.show()


# ### Optimized weights and biases

# In[67]:


# Optimized weights and biases
print('Optimized Weights: ', optWeights)
print('Optimized Biases: ', optBiases)


# ### Make predictions using the optimized weights

# In[68]:


# Forward feed with optimized weights
# Perform Forward feed and get the outputs at each layers and the inputs at each layer
a, z = forward_feed(inputs, nLayers, optWeights, optBiases, activation_func_names)
print('Outputs at each layer (including input layer):\n ',a)
print('Inputs at each layer :\n ', z)
# New Error
print('New Average Error with optimized weights:\n', MSE_loss(a[nLayers],outputOR)/batchSize)
print('Input: ',inputs)
print('Output: ',a[nLayers])
print('Expected Output: ',outputOR)


# ## Train neural network for XOR gate
# We will use the Mean Squared Error loss function (MSE_loss)

# In[69]:



# Run optimization
optWeights, optBiases, errorPlot = nn_optimize_fast(inputs, outputXOR, activation_func_names, nLayers, nEpochs=nEpochs, batchSize=batchSize, eeta=eeta, weights=weightsOriginal, biases=biasesOriginal, errorFunc=MSE_loss, gradErrorFunc=MSE_loss_grad,miniterEpoch=1,batchProgressBar=False,miniterBatch=100)


# ### Visualize the training loss

# In[70]:


# Plot the error vs epochs
plt.plot(errorPlot)
plt.yscale('log')
plt.show()


# ### Optimized weights and biases

# In[71]:


# Optimized weights and biases
print('Optimized Weights: ', optWeights)
print('Optimized Biases: ', optBiases)


# ### Make predictions using the optimized weights

# In[72]:


# Forward feed with optimized weights
# Perform Forward feed and get the outputs at each layers and the inputs at each layer
a, z = forward_feed(inputs, nLayers, optWeights, optBiases, activation_func_names)
print('Outputs at each layer (including input layer):\n ',a)
print('Inputs at each layer :\n ', z)
# New Error
print('New Average Error with optimized weights:\n', MSE_loss(a[nLayers],outputXOR)/batchSize)
print('Input: ',inputs)
print('Output: ',a[nLayers])
print('Expected Output: ',outputXOR)

