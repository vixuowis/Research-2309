from typing import *
from transformers import AutoModelForImageClassification

def load_custom_model():
    """Load a custom model for image classification.

    Returns:
        model: The loaded custom model.
    """
    model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
    return model
