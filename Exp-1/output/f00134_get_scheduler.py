from typing import *
from transformers import get_scheduler

def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    """Creates a learning rate scheduler.

    Args:
        name (str): The name of the scheduler. Currently supported schedulers are:
            - 'linear'
            - 'cosine'
            - 'cosine_with_restarts'
            - 'polynomial'
            - 'constant'
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of warmup steps.
        num_training_steps (int): The total number of training steps.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler."""
    scheduler = None

    if name == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif name == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif name == 'cosine_with_restarts':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif name == 'polynomial':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif name == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps
        )

    return scheduler
