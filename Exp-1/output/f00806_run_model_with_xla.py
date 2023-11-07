from typing import *
import tensorflow as tf

def run_model_with_xla(model):
    """Run the given TensorFlow model with XLA optimization.

    Args:
        model (tf.keras.Model): The TensorFlow model to run.

    Returns:
        tf.Tensor: The output tensor of the model.
    """
    with tf.device('/CPU:0'):
        tf.config.optimizer.set_jit(True)
        output = model(tf.ones((1, 10)))
        return output
