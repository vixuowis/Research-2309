from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO


def classify_animal_images(image_url: str):
    """
    This function classifies images of animals into their specific categories using the 'laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft' model from Hugging Face.
    
    Parameters:
    image_url (str): The URL of the image to be classified.
    
    Returns:
    str: The predicted category of the animal in the image.
    """
    # Load the model
    classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
    
    # Define the categories
    categories = ['cat', 'dog', 'bird', 'fish']
    
    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Classify the image
    result = classifier(image, categories)
    
    # Return the predicted category
    return result[0]['label']