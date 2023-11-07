from typing import *
def from_pretrained(model_name_or_path, device_map=None, **kwargs):
    """Loads a pre-trained model from a given model_name_or_path."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map=device_map, **kwargs)
