# function_import --------------------

from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

# function_code --------------------

def identify_street_location(image_url: str, choices: list):
    """
    Identify the location of a street-level image using the Hugging Face Transformers' CLIPModel.

    Args:
        image_url (str): The URL of the street-level image.
        choices (list): A list of possible locations.

    Returns:
        str: The location with the highest probability.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')

    image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    max_prob_index = probs.argmax().item()
    return choices[max_prob_index]

# test_function_code --------------------

def test_identify_street_location():
    """
    Test the identify_street_location function.
    """
    image_url = 'https://placekitten.com/200/300'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    location = identify_street_location(image_url, choices)
    assert location in choices

    image_url = 'https://placekitten.com/200/300'
    choices = ['New York', 'Chicago', 'Boston', 'Seattle', 'Austin']

    location = identify_street_location(image_url, choices)
    assert location in choices

    image_url = 'https://placekitten.com/200/300'
    choices = ['London', 'Paris', 'Berlin', 'Rome', 'Madrid']

    location = identify_street_location(image_url, choices)
    assert location in choices

    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_street_location()