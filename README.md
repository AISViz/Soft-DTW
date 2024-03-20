# Soft-Dynamic Time Warping (Soft-DTW) for TensorFlow-Keras
[Soft-DTW](https://github.com/mblondel/soft-dtw) is a variation of Dynamic Time Wrapping. A gamma factor is introduced in the minimum function. The authors developed this distance function in python for generic use and primary implementation is available [here](https://github.com/mblondel/soft-dtw). 

This repository provides the implementation of Soft-DTW as loss function for batch processing in Keras/Tensorflow models. First, Euclidean distance matrix is calculated for whole batch at once. In the next step, each sample in the batch is traversed sequentially to calculate loss (distance). To speed up the process, [same](https://github.com/mblondel/soft-dtw/blob/master/sdtw/soft_dtw_fast.pyx) Cython function **`min(a,b,c, gamma)`** implemented by authors is used. 


# python libraries
As some python libraries are prerequisite to run this loss funciton. 

`conda install numpy scipy scikit-learn cython nose`

`python3 -m pip install tensorflow==2.10`

The execution of this loss function is tested on tf~v2. User can install `tensorflow-gpu` for executing the model on gpu.

# Setup
- Compiling the cython file.

`cython sdtw/soft_dtw_fast.pyx`

- Building the package
`python setup.py build`

`python setup.py build_ext --inplace`

# Example
```import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras

from softdtwkeras.SDTWLoss import SDTWLoss

np.random.seed(42)
    input_data = np.random.random((128, 5, 2))
    output_data = np.random.random((128, 3, 2))

input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
output_tensor = tf.convert_to_tensor(output_data, dtype=tf.float32)

# Define the model
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(5, 2)),
    layers.Dense(2),
    layers.TimeDistributed(layers.Dense(2)),
    layers.Lambda(lambda x: x[:, -3:, :])  # Select last 3 timesteps
])

lossclass_sdtw = SDTWLoss(gamma=0.5)

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=lossclass_sdtw, run_eagerly=True)

model.fit(input_tensor, output_tensor, epochs=5)
```
