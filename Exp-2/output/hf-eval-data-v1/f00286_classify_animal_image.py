from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO


def classify_animal_image(image_url):
    """
    This function classifies an image into one of the following categories: cat, dog, bird.
    It uses the Hugging Face's pipeline for image classification with the pre-trained model 'laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup'.
    
    Parameters:
    image_url (str): The URL of the image to be classified.
    
    Returns:
    str: The predicted category (cat, dog, bird) of the image.
    """
    # Load the pre-trained model
    model = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    
    # Define the class names
    class_names = 'cat, dog, bird'
    
    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Use the model to classify the image
    results = model(image, class_names=class_names)
    
    # Return the predicted category
    return results[0]['label']