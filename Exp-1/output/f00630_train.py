from typing import *
from transformers import Trainer

def train(self) -> None:
    """Finetune the model."""
    self.trainer.train()
