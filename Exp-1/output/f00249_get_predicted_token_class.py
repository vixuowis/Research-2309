from typing import *
import torch

def get_predicted_token_class(logits, id2label):
    '''
    Get the class with the highest probability, and use the model's `id2label` mapping to convert it to a text label:

    :param logits: The logits tensor
    :param id2label: The mapping from class IDs to text labels
    :return: The predicted token classes as a list of text labels
    '''
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [id2label[t.item()] for t in predictions[0]]
    return predicted_token_class
