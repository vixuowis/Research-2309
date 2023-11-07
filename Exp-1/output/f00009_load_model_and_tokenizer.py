from typing import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the pretrained model and its associated tokenizer.

    Args:
        model_name (str): The name of the pretrained model to load.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: The loaded model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
