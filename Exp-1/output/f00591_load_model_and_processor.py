from typing import *
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def load_model_and_processor(checkpoint):
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
    return image_processor, model
