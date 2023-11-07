from typing import *
from transformers import pipeline

def pipeline(model, device):
    '''
    Creates a pipeline for transcribing speech using the specified model and device.

    Args:
        model (str): The name or path of the pre-trained model to use.
        device (int): The device index to use for the model.

    Returns:
        pipeline: The pipeline object for transcribing speech.
    '''
    transcriber = pipeline(model=model, device=device)
