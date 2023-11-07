from typing import *
from accelerate import Accelerator

def backward(loss):
    '''
    Replaces the typical loss.backward() in the training loop with 🤗 Accelerate's Accelerator.backward method.

    Args:
        loss: The loss value to compute gradients from.
    '''
    accelerator.backward(loss)
