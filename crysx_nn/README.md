<div id="top"></div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Features</summary>
  <ol>
    <li>
      <a href="#Activation-Functions">Activation Functions (CPU)</a>
      <ul>
        <li><a href="#Sigmoid">Sigmoid</a></li>
      </ul>
      <ul>
        <li><a href="#Tanh">Tanh</a></li>
      </ul>
      <ul>
        <li><a href="#Tanh_offset">Tanh_offset</a></li>
      </ul>
      <ul>
        <li><a href="#ReLU">ReLU</a></li>
      </ul>
      <ul>
        <li><a href="#Softplus">Softplus</a></li>
      </ul>
      <ul>
        <li><a href="#Softmax">Softmax</a></li>
      </ul>
    <li>
      <a href="#Loss-Functions">Loss Functions (CPU)</a>
      <ul>
        <li><a href="#Mean-Absolute-Error-(MAE_loss)">Mean Absolute Error (CPU)</a></li>
        <li><a href="#Mean-Squared-Error-(MSE_loss)">Mean Squared Error (CPU)</a></li>
        <li><a href="#Binary-Cross-Entropy-(BCE_loss)">Binary Cross Entropy (CPU)</a></li>
        <li><a href="#Categorical-Cross-Entropy-(CCE_loss)">Categorical Cross Entropy (CPU)</a></li>
      </ul>
    </li>
    <li><a href="#Neural-Network-Functions">Neural Network Functions</a></li>
  </ol>
</details>

## Activation Functions
**List of available activation functions and their gradients `activation.py`**

#### Sigmoid 
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Sigmoid(inputs)
gradient_wrt_inputs = activation.Sigmoid_grad(inputs)
```
#### Tanh
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Sigmoid(inputs)
gradient_wrt_inputs = activation.Tanh_grad(inputs)
```
#### Tanh_offset
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Tanh_offset(inputs)
gradient_wrt_inputs = activation.Tanh_offset_grad(inputs)
```
#### ReLU
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.ReLU(inputs)
gradient_wrt_inputs = activation.ReLU_grad(inputs)
```
#### Softplus
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Softplus(inputs)
gradient_wrt_inputs = activation.Softplus_grad(inputs)
```
#### Softmax
```python
import crysx_nn.activation as activation
import numpy as np
# inputs should be a 2d array where the rows correspond to the samples and the columns correspond to the nodes.
inputs = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
output = activation.Softmax(inputs)
jacobian_wrt_inputs = activation.Softmax_grad(inputs) 
```
<p align="right">(<a href="#top">back to top</a>)</p>

## Loss Functions 
**List of available loss functions and their gradients `loss.py`**

#### Mean Absolute Error (MSE_loss)
```python
from crysx_nn.loss import MAE_loss, MAE_loss_grad
import numpy as np
# predictions/targets should be a 2d array where the rows correspond to the samples and the columns correspond to the output nodes.
predictions = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
targets = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
error = MAE_loss(predictions, targets)
gradient_wrt_predictions = MAE_loss_grad(predictions, targets)
```
#### Mean Squared Error (MSE_loss)
```python
from crysx_nn.loss import MSE_loss, MSE_loss_grad
import numpy as np
# predictions/targets should be a 2d array where the rows correspond to the samples and the columns correspond to the output nodes.
predictions = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
targets = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
error = MSE_loss(predictions, targets)
gradient_wrt_predictions = loss.MSE_loss_grad(predictions, targets)
```
#### Binary Cross Entropy (BCE_loss)
```python
from crysx_nn.loss import MSE_loss, MSE_loss_grad
import numpy as np
# predictions/targets should be a 2d array where the rows correspond to the samples and the columns correspond to the output nodes.
predictions = np.random.random((5,4)) # BatchSize=nSamples=5; nodes=4
targets = np.random.randn((5,4)) # BatchSize=nSamples=5; nodes=4
error = MSE_loss(predictions, targets)
gradient_wrt_predictions = loss.MSE_loss_grad(predictions, targets)
```
#### Categorical Cross Entropy (CCE_loss)

<p align="right">(<a href="#top">back to top</a>)</p>

## Neural Network Functions 
**List of available neural network functions and their gradients `network.py`**

#### Visualize

#### Optimize

#### Forward Feed

#### Backpropagation

<p align="right">(<a href="#top">back to top</a>)</p>