from typing import *
from transformers import create_optimizer

def create_optimizer(init_lr, num_train_steps, weight_decay_rate, num_warmup_steps):
    """Create an optimizer and a learning rate schedule for finetuning.

    Args:
        init_lr (float): The initial learning rate.
        num_train_steps (int): The total number of training steps.
        weight_decay_rate (float): The weight decay rate.
        num_warmup_steps (int): The number of warmup steps.

    Returns:
        optimizer: The optimizer.
        lr_schedule: The learning rate schedule.
    """
    pass
