from typing import *
from transformers import Trainer

def train_model(trainer):
    """
    Trains the model using the provided trainer.

    Args:
        trainer (Trainer): The trainer object for training the model.
    Returns:
        None
    """
    trainer.train()
