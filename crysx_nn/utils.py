act_func_dict = {'Sigmoid':activation.Sigmoid,'ReLU':activation.ReLU,'Softmax':activation.Softmax}
act_func_grad_dict = {'Sigmoid':activation.Sigmoid_grad,'ReLU':activation.ReLU_grad,'Softmax':activation.Softmax_grad}
##------------------CUPY----------------------
act_func_dict_cupy = {'Sigmoid':activation.Sigmoid_cupy,'ReLU':activation.ReLU_cupy,'Softmax':activation.Softmax_cupy}
act_func_grad_dict_cupy = {'Sigmoid':activation.Sigmoid_grad_cupy,'ReLU':activation.ReLU_grad_cupy,'Softmax':activation.Softmax_grad_cupy}