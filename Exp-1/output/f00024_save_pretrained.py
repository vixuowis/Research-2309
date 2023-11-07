from typing import *
from transformers import TFPreTrainedModel

def save_pretrained(self, save_directory: str) -> None:
    '''
    Save the model and its tokenizer to a directory.

    Args:
        save_directory (:obj:`str`): Directory to save the model to.
    '''
    self.save_pretrained(save_directory)
