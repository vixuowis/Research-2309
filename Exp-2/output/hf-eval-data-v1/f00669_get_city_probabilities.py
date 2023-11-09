from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch


def get_city_probabilities(image_url, choices):
    '''
    This function takes an image URL and a list of city names as input.
    It uses the pretrained 'geolocal/StreetCLIP' model from Hugging Face Transformers to geolocalize the image.
    It returns a dictionary with the city names as keys and their corresponding probabilities as values.
    '''
    # Load the 'geolocal/StreetCLIP' model and processor
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')

    # Fetch the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the texts and image
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)

    # Pass the input to the model
    outputs = model(**inputs)

    # Compute probabilities for different cities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    # Create a dictionary with city names and their probabilities
    city_probs = dict(zip(choices, probs))

    return city_probs