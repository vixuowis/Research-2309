from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests

def detect_objects(image_url):
    '''
    This function detects objects in an image using the DeformableDetrForObjectDetection model from Hugging Face Transformers.
    Args:
    image_url (str): The URL of the image to process.
    Returns:
    dict: The detected objects in the image.
    '''
    # Load the image from the given URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Instantiate the AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    # Instantiate the DeformableDetrForObjectDetection model
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    # Process the image
    inputs = processor(images=image, return_tensors='pt')
    # Detect objects in the image
    outputs = model(**inputs)
    return outputs