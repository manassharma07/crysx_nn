import crysx_nn.activation as activation
import crysx_nn.loss as loss
import seaborn as sn
import matplotlib.pyplot as plt

act_func_dict = {'Sigmoid':activation.Sigmoid,'ReLU':activation.ReLU,'Softmax':activation.Softmax, \
    'Tanh':activation.Tanh, 'Tanh_offset':activation.Tanh_offset, 'Identity':activation.Identity, 'Softplus':activation.Softplus}

act_func_grad_dict = {'Sigmoid':activation.Sigmoid_grad,'ReLU':activation.ReLU_grad,'Softmax':activation.Softmax_grad,\
     'Tanh':activation.Tanh_grad, 'Tanh_offset':activation.Tanh_offset_grad, 'Identity':activation.Identity_grad, 'Softplus':activation.Softplus_grad}

loss_func_dict = {'MAE':loss.MAE_loss,'MSE':loss.MSE_loss,'BCE':loss.BCE_loss, \
    'CCE':loss.CCE_loss}

loss_func_grad_dict = {'MAE':loss.MAE_loss_grad,'MSE':loss.MSE_loss_grad,'BCE':loss.BCE_loss_grad, \
    'CCE':loss.CCE_loss_grad}

import numpy as np

def compute_confusion_matrix(predictions, targets):
  '''Computes a confusion matrix using numpy for two arrays
  targets and predictions.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''

  K = len(np.unique(targets)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(targets)):
    result[targets[i]][predictions[i]] += 1

  return result

def plot_confusion_matrix(confusion_matrix, tick_labels=None, x_label='predicted type',y_label='true type'):
    plt.rcParams["figure.figsize"] = (14,10)
    if tick_labels is None:
        tick_labels = range(confusion_matrix.shape[0])
    sn.heatmap(confusion_matrix, square=True, annot=True, cbar=True, cmap = sn.cm.rocket_r, xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Confusion matrix')
    plt.show()


##------------------CUPY----------------------

act_func_dict_cupy = {'Sigmoid':activation.Sigmoid_cupy,'ReLU':activation.ReLU_cupy,'Softmax':activation.Softmax_cupy, \
    'Tanh':activation.Tanh_cupy, 'Tanh_offset':activation.Tanh_offset_cupy, 'Identity':activation.Identity_cupy, 'Softplus':activation.Softplus_cupy}
    
act_func_grad_dict_cupy = {'Sigmoid':activation.Sigmoid_grad_cupy,'ReLU':activation.ReLU_grad_cupy,'Softmax':activation.Softmax_grad_cupy, \
    'Tanh':activation.Tanh_grad_cupy, 'Tanh_offset':activation.Tanh_offset_grad_cupy, 'Identity':activation.Identity_grad_cupy, 'Softplus':activation.Softplus_grad_cupy}

loss_func_dict_cupy = {'MAE':loss.MAE_loss_cupy,'MSE':loss.MSE_loss_cupy,'BCE':loss.BCE_loss_cupy, \
    'CCE':loss.CCE_loss_cupy}

loss_func_grad_dict_cupy = {'MAE':loss.MAE_loss_grad_cupy,'MSE':loss.MSE_loss_grad_cupy,'BCE':loss.BCE_loss_grad_cupy, \
    'CCE':loss.CCE_loss_grad_cupy}


try:
    import cupy as cp                     
except ImportError:
    print('Warning: CuPy could not be imported! You can only use CPU for computations.')

def compute_confusion_matrix_cupy(predictions, targets):
  '''Computes a confusion matrix using numpy for two arrays
  targets and predictions.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''

  K = len(cp.unique(targets)) # Number of classes 
  result = cp.zeros((K, K))

  for i in range(len(targets)):
    result[targets[i]][predictions[i]] += 1

  return result