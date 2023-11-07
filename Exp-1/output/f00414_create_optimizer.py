from typing import *
from transformers import AdamWeightDecay, get_linear_schedule_with_warmup

def create_optimizer(init_lr, num_warmup_steps, num_train_steps):
    '''Create an optimizer and learning rate schedule for finetuning a model.

    Args:
        init_lr (float): The initial learning rate.
        num_warmup_steps (int): The number of warmup steps for the learning rate schedule.
        num_train_steps (int): The total number of training steps.

    Returns:
        optimizer: The optimizer.
        schedule: The learning rate schedule.'''
    optimizer = AdamWeightDecay(learning_rate=init_lr, weight_decay_rate=0.01, epsilon=1e-6, exclude_from_weight_decay=['layer_norm', 'bias'])
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    return optimizer, schedule
