from typing import *
from transformers import pipeline

def pipeline(model: str) -> Callable[..., Any]:
    # This function creates a pipeline for NLP tasks using the specified model.
    # It returns a callable object that can be used to classify text.
    return pipeline(model)
