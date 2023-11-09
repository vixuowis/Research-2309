# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def get_location_probabilities(image_url: str, choices: list) -> dict:
    """
    This function uses the pretrained 'geolocal/StreetCLIP' model to classify an image into one of the given city options.
    The function returns a dictionary with city names as keys and their corresponding probabilities as values.

    Args:
        image_url (str): The URL of the image to be classified.
        choices (list): A list of city names to classify the image into.

    Returns:
        dict: A dictionary with city names as keys and their corresponding probabilities as values.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs_dict = {choice: prob for choice, prob in zip(choices, probs.tolist()[0])}
    return probs_dict

# test_function_code --------------------

def test_get_location_probabilities():
    """
    This function tests the get_location_probabilities function.
    It uses a sample image URL and a list of city choices.
    The function asserts that the output is a dictionary and that the sum of the probabilities is approximately 1.
    """
    image_url = 'https://example.com/potential_location_image.jpg'
    choices = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    probs_dict = get_location_probabilities(image_url, choices)
    assert isinstance(probs_dict, dict)
    assert abs(sum(probs_dict.values()) - 1) < 1e-6

# call_test_function_code --------------------

test_get_location_probabilities()