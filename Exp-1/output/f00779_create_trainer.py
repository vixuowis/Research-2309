from typing import *
from transformers import Trainer

def create_trainer(model, training_args, train_dataset, optimizers):
    '''
    This function creates a trainer object for training a model.

    Args:
        model (object): The model to be trained.
        training_args (object): The training arguments.
        train_dataset (object): The training dataset.
        optimizers (tuple): A tuple of optimizers to be used for training.

    Returns:
        trainer (object): The trainer object.
    '''
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, optimizers=optimizers)
    return trainer
