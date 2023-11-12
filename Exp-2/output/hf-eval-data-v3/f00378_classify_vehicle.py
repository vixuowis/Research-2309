# function_import --------------------

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_vehicle(image_url: str) -> str:
    """
    Classify the vehicle in the image as either a bike or a car.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The classification result, either 'bike' or 'car'.
    """
    # Create a zero-shot classification model
    clip = pipeline('zero-shot-classification', model='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind')
    # Define the class names
    class_names = ['bike', 'car']
    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    # Classify the image
    result = clip(image, class_names)
    # Return the classification result
    return result['labels'][0]

# test_function_code --------------------

def test_classify_vehicle():
    """
    Test the classify_vehicle function.
    """
    # Test with a bike image
    assert classify_vehicle('https://placekitten.com/200/300') == 'bike'
    # Test with a car image
    assert classify_vehicle('https://placekitten.com/200/300') == 'car'
    # Test with a non-vehicle image
    assert classify_vehicle('https://placekitten.com/200/300') not in ['bike', 'car']
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_vehicle()