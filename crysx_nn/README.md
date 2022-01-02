### List of available activation functions and their gradients

* Sigmoid 
```python
import crysx_nn.activation as activation
import numpy as np
inputs = np.random.randn((5,4))
output = activation.Sigmoid(inputs)
```
* Tanh
* Tanh_offset
* ReLU
* Softplus
* Softmax

### List of available loss functions and their gradients

* Mean Absolute Error
* Mean Squared Error
* Binary Cross Entropy
* Categorical Cross Entropy
