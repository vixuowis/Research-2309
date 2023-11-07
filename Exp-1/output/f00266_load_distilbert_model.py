from typing import *
from transformers import TFAutoModelForQuestionAnswering

def load_distilbert_model(model_name):
    '''
    Loads and returns a DistilBERT model for question answering.

    Args:
        model_name (str): The name of the DistilBERT model to load.

    Returns:
        TFAutoModelForQuestionAnswering: The loaded DistilBERT model.
    '''
    model = TFAutoModelForQuestionAnswering(model_name)
    return model
