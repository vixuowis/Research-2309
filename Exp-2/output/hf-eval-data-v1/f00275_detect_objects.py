from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

def detect_objects(url):
    '''
    This function detects objects in an image from a URL using the pre-trained DETR model from Hugging Face Transformers.
    
    Parameters:
    url (str): The URL of the image.
    
    Returns:
    outputs (dict): The detected objects and their confidence scores.
    '''
    # Open the image from the URL
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Load the pre-trained DETR model and the image processor
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')
    
    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')
    
    # Pass the preprocessed image to the model
    outputs = model(**inputs)
    
    return outputs