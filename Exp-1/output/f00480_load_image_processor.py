from typing import *
from transformers import AutoImageProcessor

def load_image_processor(checkpoint: str) -> AutoImageProcessor:
    
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    return image_processor
