from typing import *
from transformers import AutoImageProcessor

def preprocess_data(checkpoint: str) -> AutoImageProcessor:
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return image_processor
