import crysx_nn.activation as activation

act_func_dict = {'Sigmoid':activation.Sigmoid,'ReLU':activation.ReLU,'Softmax':activation.Softmax, \
    'Tanh':activation.Tanh, 'Tanh_offset':activation.Tanh_offset, 'Identity':activation.Identity, 'Softplus':activation.Softplus}

act_func_grad_dict = {'Sigmoid':activation.Sigmoid_grad,'ReLU':activation.ReLU_grad,'Softmax':activation.Softmax_grad,\
     'Tanh':activation.Tanh_grad, 'Tanh_offset':activation.Tanh_offset_grad, 'Identity':activation.Identity_grad, 'Softplus':activation.Softplus_grad}

##------------------CUPY----------------------

act_func_dict_cupy = {'Sigmoid':activation.Sigmoid_cupy,'ReLU':activation.ReLU_cupy,'Softmax':activation.Softmax_cupy, \
    'Tanh':activation.Tanh_cupy, 'Tanh_offset':activation.Tanh_offset_cupy, 'Identity':activation.Identity_cupy, 'Softplus':activation.Softplus_cupy}
    
act_func_grad_dict_cupy = {'Sigmoid':activation.Sigmoid_grad_cupy,'ReLU':activation.ReLU_grad_cupy,'Softmax':activation.Softmax_grad_cupy, \
    'Tanh':activation.Tanh_grad_cupy, 'Tanh_offset':activation.Tanh_offset_grad_cupy, 'Identity':activation.Identity_grad_cupy, 'Softplus':activation.Softplus_grad_cupy}