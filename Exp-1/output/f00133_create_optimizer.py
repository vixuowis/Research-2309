from typing import *
from torch.optim import AdamW

def create_optimizer(model, learning_rate):
    """Create an optimizer using AdamW and set the learning rate.

    Args:
        model: The model to be optimized.
        learning_rate: The learning rate for the optimizer.

    Returns:
        The optimizer.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer
