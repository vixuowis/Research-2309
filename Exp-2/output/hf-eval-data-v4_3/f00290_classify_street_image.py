# requirements_file --------------------

import subprocess

requirements = ["transformers", "PILlow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_street_image(image_url, choices):
    """
    Classify a street-level image and identify the location among a list of choices.

    Args:
        image_url (str): The URL of the image to classify.
        choices (list of str): A list of location names to choose from.

    Returns:
        str: The location name with the highest probability.

    Raises:
        ValueError: If `image_url` is not reachable or `choices` is empty.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Image cannot be retrieved from URL: {e}')

    if not choices:
        raise ValueError('Choices list cannot be empty')
    
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    highest_prob_index = probs.argmax()
    return choices[highest_prob_index]

# test_function_code --------------------

def test_classify_street_image():
    print("Testing started.")
    image_url = 'https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    # Testing case 1: Working example
    print("Testing case [1/3] started.")
    result = classify_street_image(image_url, choices)
    assert result in choices, f"Test case [1/3] failed: {result} not in choices"

    # Testing case 2: Invalid URL
    print("Testing case [2/3] started.")
    try:
        classify_street_image('invalid_url', choices)
        assert False, "Test case [2/3] failed: ValueError not raised for invalid URL"
    except ValueError:
        pass

    # Testing case 3: Empty choices
    print("Testing case [3/3] started.")
    try:
        classify_street_image(image_url, [])
        assert False, "Test case [3/3] failed: ValueError not raised for empty choices"
    except ValueError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_street_image()