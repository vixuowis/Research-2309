from typing import *
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


def load_model_and_processor(checkpoint: str) -> Tuple[AutoModelForZeroShotImageClassification, AutoProcessor]:
    """
    Load the model and associated processor from a checkpoint on the Hugging Face Hub.

    Args:
        checkpoint (str): The checkpoint name or path on the Hugging Face Hub.

    Returns:
        Tuple[AutoModelForZeroShotImageClassification, AutoProcessor]: The loaded model and processor.
    """
    model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor
