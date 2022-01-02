### List of available activation functions and their gradients `activation.py`

* Sigmoid 
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Sigmoid(inputs)
gradient_wrt_inputs = activation.Sigmoid_grad(inputs)

```
* Tanh
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Sigmoid(inputs)
gradient_wrt_inputs = activation.Tanh_grad(inputs)
```
* Tanh_offset
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Tanh_offset(inputs)
gradient_wrt_inputs = activation.Tanh_offset_grad(inputs)
```
* ReLU
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.ReLU(inputs)
gradient_wrt_inputs = activation.ReLU_grad(inputs)
```
* Softplus
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Softplus(inputs)
gradient_wrt_inputs = activation.Softplus_grad(inputs)
```
* Softmax
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Softmax(inputs)
jacobian_wrt_inputs = activation.Softmax_grad(inputs) 
```

### List of available loss functions and their gradients `loss.py`

* Mean Absolute Error
* Mean Squared Error
* Binary Cross Entropy
* Categorical Cross Entropy
