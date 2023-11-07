from typing import *
from transformers import TFAutoModelForMultipleChoice

def load_bert_model():
    """Load BERT model for multiple choice tasks.

    Returns:
        model (TFAutoModelForMultipleChoice): The loaded BERT model.
    """
    model = TFAutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
    return model
