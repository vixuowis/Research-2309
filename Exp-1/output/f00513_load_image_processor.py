from typing import *
from transformers import AutoImageProcessor

def load_image_processor(checkpoint: str) -> AutoImageProcessor:
    """Load a SegFormer image processor to prepare the images and annotations for the model.

    Args:
        - checkpoint (str): The checkpoint name or path.

    Returns:
        - AutoImageProcessor: The loaded image processor.
    """
    return AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
