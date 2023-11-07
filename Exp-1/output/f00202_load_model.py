from typing import *
from transformers import AutoModelForSequenceClassification

def load_model(num_labels, id2label, label2id):
    """
    Load DistilBERT with AutoModelForSequenceClassification along with the number of expected labels, and the label mappings
    :param num_labels: int, the number of expected labels
    :param id2label: dict, mapping of label ids to labels
    :param label2id: dict, mapping of labels to label ids
    :return: model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    return model
