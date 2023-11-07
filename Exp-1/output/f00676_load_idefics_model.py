from typing import *
import torch

from transformers import IdeficsForVisionText2Text, AutoProcessor

def load_idefics_model(checkpoint: str) -> Tuple[AutoProcessor, IdeficsForVisionText2Text]:
    """Load the IDEFICS model and processor from the given checkpoint.

    Args:
        checkpoint (str): The path to the checkpoint directory.

    Returns:
        Tuple[AutoProcessor, IdeficsForVisionText2Text]: A tuple containing the loaded processor and model.
    """
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")

    return processor, model
