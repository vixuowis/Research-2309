from typing import *
from transformers import FlaxDistilBertForSequenceClassification

def from_pretrained(model_path: str, from_pt: bool) -> FlaxDistilBertForSequenceClassification:
    """Load a pretrained FlaxDistilBertForSequenceClassification model from a given path.

    Args:
        model_path (str): The path to the pretrained model.
        from_pt (bool): Whether to load from a PyTorch checkpoint.

    Returns:
        FlaxDistilBertForSequenceClassification: The loaded pretrained model."""
    flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(model_path, from_pt=from_pt)
