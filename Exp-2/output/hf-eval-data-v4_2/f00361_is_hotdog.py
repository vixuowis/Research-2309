# requirements_file --------------------

!pip install -U transformers pillow requests

# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def is_hotdog(image_url):
    """
    Determines whether the image located at the provided URL is a hotdog or not.

    Args:
        image_url: str, a URL pointing to an image file.

    Returns:
        A string 'hotdog' or 'not hotdog' based on the image classification.

    Raises:
        ValueError: If the image_url does not lead to a valid image.
    """
    image_classifier = pipeline('image-classification', model='julien-c/hotdog-not-hotdog')

    try:
        # Load the image from the provided URL
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure the request was successful
        img = Image.open(BytesIO(response.content))
    except (requests.RequestException, IOError):
        raise ValueError('Invalid image URL or failed to load image.')

    # Classify the image using the hotdog-not-hotdog classifier
    result = image_classifier(img)
    prediction = result[0]['label']

    return 'hotdog' if 'hot-dog' in prediction else 'not hotdog'

# test_function_code --------------------

def test_is_hotdog():
    print("Testing started.")
    image_hotdog_url = 'https://example.com/hotdog.jpg'
    image_not_hotdog_url = 'https://example.com/not_hotdog.jpg'
    image_invalid_url = 'https://invalid_url'

    # Test case 1: Hotdog image classification
    print("Testing case [1/3] started.")
    assert is_hotdog(image_hotdog_url) == 'hotdog', "Test case [1/3] failed: The image should be classified as a hotdog."

    # Test case 2: Not hotdog image classification
    print("Testing case [2/3] started.")
    assert is_hotdog(image_not_hotdog_url) == 'not hotdog', "Test case [2/3] failed: The image should be classified as not a hotdog."

    # Test case 3: Invalid image URL
    print("Testing case [3/3] started.")
    try:
        is_hotdog(image_invalid_url)
        assert False, "Test case [3/3] failed: Exception for invalid URL was not raised."
    except ValueError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_is_hotdog()