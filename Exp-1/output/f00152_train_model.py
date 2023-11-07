from typing import *
from transformers import Trainer

def train_model(model):
    trainer = Trainer(model=model, ...)
    trainer.train()
    return None
