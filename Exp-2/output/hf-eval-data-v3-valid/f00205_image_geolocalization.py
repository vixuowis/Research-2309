# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def image_geolocalization(url: str, choices: list):
    """
    This function uses a pretrained CLIP model to identify the location of a given image.

    Args:
        url (str): The URL of the image to be geolocalized.
        choices (list): A list of possible choices for the location of the image.

    Returns:
        dict: A dictionary with the location choices as keys and their corresponding probabilities as values.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {choice: prob for choice, prob in zip(choices, probs.tolist()[0])}

# test_function_code --------------------

def test_image_geolocalization():
    url = 'https://placekitten.com/200/300'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']
    result = image_geolocalization(url, choices)
    assert isinstance(result, dict)
    assert len(result) == len(choices)
    assert all(isinstance(choice, str) for choice in result.keys())
    assert all(isinstance(prob, float) for prob in result.values())
    assert abs(sum(result.values()) - 1) < 1e-6
    return 'All Tests Passed'

# call_test_function_code --------------------

test_image_geolocalization()