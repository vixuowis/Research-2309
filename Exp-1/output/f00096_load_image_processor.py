from typing import *
from transformers import AutoImageProcessor

def load_image_processor(model_name: str) -> AutoImageProcessor:
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    return image_processor
