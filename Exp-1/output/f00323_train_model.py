from typing import *
from transformers import AutoModelForMaskedLM

def train_model():
    """Train the model using the `Trainer` class.

    Args:
        model: The pretrained model to train.

    Returns:
        The trained model."""
    trainer = Trainer(model)
    trainer.train()

    return model
