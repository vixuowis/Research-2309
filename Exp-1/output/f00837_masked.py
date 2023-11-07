from typing import *
from transformers import pipeline

def masked(text):
    # This function predicts a masked token in a sequence
    # Args:
    #   text (str): The input text with a masked token
    # Returns:
    #   list: A list of dictionaries containing the predicted token, its score, and the updated sequence
    fill_mask = pipeline(task='fill-mask')
    preds = fill_mask(text, top_k=1)
    preds = [
        {
            'score': round(pred['score'], 4),
            'token': pred['token'],
            'token_str': pred['token_str'],
            'sequence': pred['sequence'],
        }
        for pred in preds
    ]
    return preds
