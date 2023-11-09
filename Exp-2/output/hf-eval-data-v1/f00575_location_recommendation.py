from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel


def location_recommendation(image_url, choices):
    '''
    This function uses the pretrained 'geolocal/StreetCLIP' model to classify images of potential store locations
    and returns the probabilities for each city option.
    
    Parameters:
    image_url (str): The URL of the image to be classified.
    choices (list): A list of city options to classify images.
    
    Returns:
    probs (tensor): The probabilities for each city option.
    '''
    # Load the pretrained 'geolocal/StreetCLIP' model
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    # Instantiate a processor with the same pretrained 'geolocal/StreetCLIP' model
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    # Open the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Process the text (city options) and images
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    # Compute the logits and probabilities for each city option
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs