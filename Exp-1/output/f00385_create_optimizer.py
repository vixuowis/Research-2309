from typing import *
from transformers import create_optimizer, AdamWeightDecay

def create_optimizer(learning_rate: float, weight_decay_rate: float) -> tf.keras.optimizers.Optimizer:
    """Create an optimizer with a learning rate schedule and weight decay.

    Args:
        learning_rate (float): The initial learning rate for the optimizer.
        weight_decay_rate (float): The weight decay rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: The optimizer with the learning rate schedule and weight decay.
    """
    optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay_rate)
    return optimizer
