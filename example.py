import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras

from softdtwkeras.SDTWLoss import SDTWLoss

if __name__ == '__main__':
    import time
    start_time = time.time()
    # Generate random input and output data
    np.random.seed(42)
    input_data = np.random.random((32, 5, 2))
    output_data = np.random.random((32, 3, 2))

    # Convert to tensors to enable GPU processing
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

    end_time = time.time()
    execution_time_ms = (end_time - start_time)
    print(f"Execution time: {execution_time_ms} seconds")