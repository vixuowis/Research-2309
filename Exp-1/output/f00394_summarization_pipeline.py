from typing import *
from transformers import pipeline

def summarization_pipeline(model_name):
    """Instantiates a pipeline for summarization with the specified model.

    Args:
        model_name (str): The name or path of the model to use for summarization.

    Returns:
        Pipeline: A pipeline object for summarization.
    """
    summarizer = pipeline("summarization", model=model_name)
    return summarizer
