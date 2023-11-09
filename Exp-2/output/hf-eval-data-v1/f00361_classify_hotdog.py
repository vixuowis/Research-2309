from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

def classify_hotdog(image_url):
    """
    This function takes an image URL as input, loads the image, and uses the 'julien-c/hotdog-not-hotdog' model
    to classify whether the image is of a hotdog or not.
    
    Parameters:
    image_url (str): The URL of the image to classify.
    
    Returns:
    str: The classification result ('hotdog' or 'not hotdog').
    """
    # Load the image classification model
    image_classifier = pipeline('image-classification', model='julien-c/hotdog-not-hotdog')
    
    # Load the image from the provided URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Classify the image using the hotdog-not-hotdog classifier
    result = image_classifier(img)
    prediction = result[0]['label']
    
    return prediction