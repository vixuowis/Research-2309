from typing import *
from transformers import DistilBertForSequenceClassification

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_model_path):
    """Converts a TensorFlow checkpoint to a PyTorch checkpoint.

    Args:
        tf_checkpoint_path (str): The path to the TensorFlow checkpoint file.
        pytorch_model_path (str): The path to save the converted PyTorch checkpoint.
    """
    pt_model = DistilBertForSequenceClassification.from_pretrained(tf_checkpoint_path, from_tf=True)
    pt_model.save_pretrained(pytorch_model_path)
