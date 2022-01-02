## List of available activation functions and their gradients `activation.py`

* **Sigmoid** 
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Sigmoid(inputs)
gradient_wrt_inputs = activation.Sigmoid_grad(inputs)
```
* **Tanh**
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Sigmoid(inputs)
gradient_wrt_inputs = activation.Tanh_grad(inputs)
```
* **Tanh_offset**
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Tanh_offset(inputs)
gradient_wrt_inputs = activation.Tanh_offset_grad(inputs)
```
* **ReLU**
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.ReLU(inputs)
gradient_wrt_inputs = activation.ReLU_grad(inputs)
```
* **Softplus**
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Softplus(inputs)
gradient_wrt_inputs = activation.Softplus_grad(inputs)
```
* **Softmax**
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Softmax(inputs)
jacobian_wrt_inputs = activation.Softmax_grad(inputs) 
```

## List of available loss functions and their gradients `loss.py`

* **Mean Absolute Error (MSE_loss)**
```python
from crysx_nn.loss import MAE_loss, MAE_loss_grad
import numpy as np
# predictions/targets should be a 2d array where the rows correspond to the samples and the columns correspond to the output nodes.
predictions = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
targets = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
error = MAE_loss(predictions, targets)
gradient_wrt_predictions = MAE_loss_grad(predictions, targets)
```
* **Mean Squared Error (MSE_loss)**
```python
from crysx_nn.loss import MSE_loss, MSE_loss_grad
import numpy as np
# predictions/targets should be a 2d array where the rows correspond to the samples and the columns correspond to the output nodes.
predictions = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
targets = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
error = MSE_loss(predictions, targets)
gradient_wrt_predictions = loss.MSE_loss_grad(predictions, targets)
```
* **Binary Cross Entropy (BCE_loss)**
```python
from crysx_nn.loss import MSE_loss, MSE_loss_grad
import numpy as np
# predictions/targets should be a 2d array where the rows correspond to the samples and the columns correspond to the output nodes.
predictions = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
targets = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
error = MSE_loss(predictions, targets)
gradient_wrt_predictions = loss.MSE_loss_grad(predictions, targets)
```
* **Categorical Cross Entropy (CCE_loss)**
