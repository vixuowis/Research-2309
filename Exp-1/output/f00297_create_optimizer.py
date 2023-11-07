from typing import *
from transformers import create_optimizer, AdamWeightDecay

def create_optimizer(learning_rate: float, weight_decay_rate: float, epsilon: float = 1e-8) -> tf.keras.optimizers.Optimizer:
    """Create an optimizer with weight decay and epsilon parameters.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        weight_decay_rate (float): The weight decay rate for the optimizer.
        epsilon (float, optional): A small constant for numerical stability. Defaults to 1e-8.

    Returns:
        tf.keras.optimizers.Optimizer: The optimizer with weight decay and epsilon parameters.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    optimizer = tfa.optimizers.weight_decay_optimization.WeightDecayOptimization(
        optimizer=optimizer,
        weight_decay_rate=weight_decay_rate,
        name="AdamW"
    )
    return optimizer
