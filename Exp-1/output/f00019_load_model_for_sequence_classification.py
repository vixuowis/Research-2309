from typing import *
from transformers import TFAutoModelForSequenceClassification

def load_model_for_sequence_classification(model_name: str) -> TFAutoModelForSequenceClassification:
    """Load a pretrained TFAutoModelForSequenceClassification model

    Args:
        model_name (str): The name of the pretrained model

    Returns:
        TFAutoModelForSequenceClassification: The loaded pretrained model
    """
    return TFAutoModelForSequenceClassification.from_pretrained(model_name)
