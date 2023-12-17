# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def is_hotdog(image_url):
    """
    Determine whether an image at a given URL is a hotdog.

    Parameters:
    - image_url (str): The URL of the image to be classified.

    Returns:
    - str: 'hotdog' if the image is classified as a hotdog, otherwise 'not hotdog'.
    """
    # Load image classification model
    image_classifier = pipeline('image-classification', model='julien-c/hotdog-not-hotdog')

    # Load the image from the provided URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Classify the image using the hotdog-not-hotdog classifier
    result = image_classifier(img)
    prediction = result[0]['label']

    return prediction

# test_function_code --------------------

def test_is_hotdog():
    print("Testing is_hotdog function")

    # Example image URLs
    hotdog_image_url = 'https://example.com/hotdog.jpg'
    not_hotdog_image_url = 'https://example.com/not_hotdog.jpg'

    # Test case 1: Image is a hotdog
    print("Testing with a hotdog image.")
    result = is_hotdog(hotdog_image_url)
    assert result == 'hotdog', f"Test case failed: expected 'hotdog', got {result}"

    # Test case 2: Image is not a hotdog
    print("Testing with a not-hotdog image.")
    result = is_hotdog(not_hotdog_image_url)
    assert result == 'not hotdog', f"Test case failed: expected 'not hotdog', got {result}"

    print("Testing finished.")

# Run the test function
test_is_hotdog()