from typing import *
from transformers import AutoModel

def from_pretrained(model_name_or_path, revision=None, **kwargs):
    '''
    Loads a model from a pretrained model_name_or_path. If a revision is specified, the model will be loaded from that specific revision.
    '''
    if revision is not None:
        model_name_or_path = f'{model_name_or_path}@{revision}'

    # rest of the implementation

