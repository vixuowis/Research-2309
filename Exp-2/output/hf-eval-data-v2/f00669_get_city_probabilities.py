# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

# function_code --------------------

def get_city_probabilities(image_url: str, city_choices: list) -> dict:
    """
    This function takes an image URL and a list of city names as input, and returns a dictionary with the city names as keys and their corresponding probabilities as values.
    The probabilities are computed using the pretrained 'geolocal/StreetCLIP' model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to be geolocalized.
        city_choices (list): A list of city names to which the image is to be geolocalized.

    Returns:
        dict: A dictionary with city names as keys and their corresponding probabilities as values.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=city_choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]
    city_probs = dict(zip(city_choices, probs))
    return city_probs

# test_function_code --------------------

def test_get_city_probabilities():
    """
    This function tests the 'get_city_probabilities' function by using a sample image URL and a list of city names.
    The function asserts that the output is a dictionary and that the sum of the probabilities is approximately 1.
    """
    image_url = 'https://path_to_your_image.com/image.jpg'
    city_choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']
    city_probs = get_city_probabilities(image_url, city_choices)
    assert isinstance(city_probs, dict), 'Output should be a dictionary.'
    assert abs(sum(city_probs.values()) - 1) < 1e-6, 'Probabilities should sum up to approximately 1.'

# call_test_function_code --------------------

test_get_city_probabilities()