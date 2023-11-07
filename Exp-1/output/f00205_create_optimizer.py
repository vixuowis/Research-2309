from typing import *
from transformers import create_optimizer
import tensorflow as tf

def create_optimizer(init_lr, num_warmup_steps, num_train_steps):
    """Creates an optimizer and a learning rate schedule.

    Args:
        init_lr (float): The initial learning rate.
        num_warmup_steps (int): The number of warmup steps.
        num_train_steps (int): The total number of training steps.

    Returns:
        optimizer (tf.keras.optimizers.Optimizer): The optimizer.
        schedule (tf.keras.optimizers.schedules.LearningRateSchedule): The learning rate schedule.
    """
    pass
