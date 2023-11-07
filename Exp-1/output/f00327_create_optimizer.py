from typing import *
from transformers import create_optimizer, AdamWeightDecay

def create_optimizer(learning_rate: float, weight_decay_rate: float) -> tf.keras.optimizers.Optimizer:
    
    Create an optimizer with weight decay.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        weight_decay_rate (float): The weight decay rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: The optimizer with weight decay.
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer)
    optimizer = tf.keras.optimizers.get(learning_rate=learning_rate, weight_decay_rate=weight_decay_rate)
