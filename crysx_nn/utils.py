import crysx_nn.activation as activation
import crysx_nn.loss as loss

act_func_dict = {'Sigmoid':activation.Sigmoid,'ReLU':activation.ReLU,'Softmax':activation.Softmax, \
    'Tanh':activation.Tanh, 'Tanh_offset':activation.Tanh_offset, 'Identity':activation.Identity, 'Softplus':activation.Softplus}

act_func_grad_dict = {'Sigmoid':activation.Sigmoid_grad,'ReLU':activation.ReLU_grad,'Softmax':activation.Softmax_grad,\
     'Tanh':activation.Tanh_grad, 'Tanh_offset':activation.Tanh_offset_grad, 'Identity':activation.Identity_grad, 'Softplus':activation.Softplus_grad}

loss_func_dict = {'MAE':loss.MAE_loss,'MSE':loss.MSE_loss,'BCE':loss.BCE_loss, \
    'CCE':loss.CCE_loss}

loss_func_grad_dict = {'MAE':loss.MAE_loss_grad,'MSE':loss.MSE_loss_grad,'BCE':loss.BCE_loss_grad, \
    'CCE':loss.CCE_loss_grad}

##------------------CUPY----------------------

act_func_dict_cupy = {'Sigmoid':activation.Sigmoid_cupy,'ReLU':activation.ReLU_cupy,'Softmax':activation.Softmax_cupy, \
    'Tanh':activation.Tanh_cupy, 'Tanh_offset':activation.Tanh_offset_cupy, 'Identity':activation.Identity_cupy, 'Softplus':activation.Softplus_cupy}
    
act_func_grad_dict_cupy = {'Sigmoid':activation.Sigmoid_grad_cupy,'ReLU':activation.ReLU_grad_cupy,'Softmax':activation.Softmax_grad_cupy, \
    'Tanh':activation.Tanh_grad_cupy, 'Tanh_offset':activation.Tanh_offset_grad_cupy, 'Identity':activation.Identity_grad_cupy, 'Softplus':activation.Softplus_grad_cupy}

loss_func_dict_cupy = {'MAE':loss.MAE_loss_cupy,'MSE':loss.MSE_loss_cupy,'BCE':loss.BCE_loss_cupy, \
    'CCE':loss.CCE_loss_cupy}

loss_func_grad_dict_cupy = {'MAE':loss.MAE_loss_grad_cupy,'MSE':loss.MSE_loss_grad_cupy,'BCE':loss.BCE_loss_grad_cupy, \
    'CCE':loss.CCE_loss_grad_cupy}