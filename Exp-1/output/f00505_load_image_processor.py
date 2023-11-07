from typing import *
from transformers import AutoImageProcessor

def load_image_processor(model_name):
    # Load an image processor to preprocess the image and return the `input` as TensorFlow tensors:
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="tf")
