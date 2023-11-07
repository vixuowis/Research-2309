from typing import *
from transformers import BertForSequenceClassification

def get_predicted_label(logits, id2label):
    """Get the predicted label with the highest probability, and use the model's `id2label` mapping to convert it to a label:

    Args:
        logits (torch.Tensor): The output logits from the model.
        id2label (Dict[int, str]): A dictionary mapping label indices to label strings.

    Returns:
        str: The predicted label."""
    return id2label[logits.argmax(-1).item()]
