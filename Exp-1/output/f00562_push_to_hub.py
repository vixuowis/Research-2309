from typing import *
from transformers import Trainer
def push_to_hub() -> None:
    '''
    Pushes the final model to the Hugging Face Hub.

    If you have set `push_to_hub` to `True` in the `training_args`, the training checkpoints are pushed to the
    Hugging Face Hub. Upon training completion, push the final model to the Hub as well by calling the [`~transformers.Trainer.push_to_hub`] method.
    '''
    trainer.push_to_hub()
