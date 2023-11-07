from typing import *
from transformers import pipeline

def run_image_classification(model_name, image):
    classifier = pipeline("image-classification", model=model_name)
    return classifier(image)
