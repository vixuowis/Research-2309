from typing import *
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

def AutoModelForImageClassification.from_pretrained(checkpoint, num_labels, id2label, label2id):
    """Load ViT with AutoModelForImageClassification. Specify the number of labels along with the number of expected labels, and the label mappings."""
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
