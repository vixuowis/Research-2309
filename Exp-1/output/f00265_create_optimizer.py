from typing import *
from transformers import AdamW, get_linear_schedule_with_warmup
import tensorflow as tf

def create_optimizer(init_lr, num_warmup_steps, num_train_steps):
    '''
    Create an optimizer and a learning rate schedule for finetuning a model.

    Args:
        init_lr (float): The initial learning rate.
        num_warmup_steps (int): The number of warmup steps.
        num_train_steps (int): The total number of training steps.

    Returns:
        optimizer (tf.keras.optimizers.Optimizer): The optimizer.
        schedule (tf.keras.optimizers.schedules.LearningRateSchedule): The learning rate schedule.
    '''
    optimizer = AdamW(learning_rate=init_lr, epsilon=1e-8)
    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )
    return optimizer, schedule
