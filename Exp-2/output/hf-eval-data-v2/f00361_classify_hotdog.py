# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def classify_hotdog(image_url):
    """
    Classify whether the image at the given URL is a hotdog or not using the 'julien-c/hotdog-not-hotdog' model.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        str: The classification result, either 'hotdog' or 'not hotdog'.
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

# test_function_code --------------------

def test_classify_hotdog():
    """
    Test the classify_hotdog function with a known hotdog image and a known not-hotdog image.
    """
    hotdog_url = 'https://example.com/hotdog.jpg'
    not_hotdog_url = 'https://example.com/not_hotdog.jpg'

    assert classify_hotdog(hotdog_url) == 'hotdog'
    assert classify_hotdog(not_hotdog_url) == 'not hotdog'

# call_test_function_code --------------------

test_classify_hotdog()