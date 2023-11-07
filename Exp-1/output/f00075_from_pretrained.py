from typing import *
from transformers import TFAutoModelForTokenClassification

def from_pretrained(model_name_or_path, *model_args, **kwargs):
    """Loads a pre-trained model from a given model_name_or_path."""
    model = TFAutoModelForTokenClassification.from_pretrained(model_name_or_path)
