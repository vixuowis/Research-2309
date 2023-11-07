from typing import *
from transformers import ViltForQuestionAnswering

def train_model(model_checkpoint, id2label, label2id):
    """Train the model

    Parameters:
    - model_checkpoint (str): The path or name of the pre-trained model checkpoint
    - id2label (dict): A dictionary mapping label IDs to their corresponding labels
    - label2id (dict): A dictionary mapping labels to their corresponding label IDs

    Returns:
    - model (ViltForQuestionAnswering): The trained model
    """
    model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)

