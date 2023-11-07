from typing import *
from transformers import pipeline

def image_classification(image_url):
    # Function to perform image classification
    # Args:
    #     image_url (str): URL of the image to classify
    # Returns:
    #     list: List of dictionaries containing the predicted labels and scores
    classifier = pipeline(task='image-classification')
    preds = classifier(image_url)
    preds = [{'score': round(pred['score'], 4), 'label': pred['label']} for pred in preds]
    return preds

