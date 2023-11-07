from typing import *
from transformers import create_optimizer

def create_optimizer(init_lr, num_train_steps, weight_decay_rate, num_warmup_steps=0):
    ...
    return optimizer, lr_schedule
