from typing import *
from transformers import TFAutoModelForSeq2SeqLM

def load_model(checkpoint):
    '''
    Load T5 model from a checkpoint.

    Args:
        checkpoint (str): The path or name of the checkpoint to load.

    Returns:
        TFAutoModelForSeq2SeqLM: The loaded T5 model.
    '''
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return model
