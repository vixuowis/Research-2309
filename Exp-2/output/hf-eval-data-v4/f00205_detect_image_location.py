# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def detect_image_location(image_url, location_choices):
    """
    Detect the geographical location of an image using a pretrained CLIP model.

    Args:
        image_url (str): URL of the image to be analyzed.
        location_choices (list): A list of possible locations.

    Returns:
        dict: A dictionary with location probabilities.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=location_choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    location_probs = {location: float(prob) for location, prob in zip(location_choices, probs[0])}
    return location_probs

# test_function_code --------------------

def test_detect_image_location():
    print("Testing started.")
    # Sample image and location choices
    image_url = 'https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg'
    location_choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    # Test if the function returns a dictionary
    print("Testing case [1/1] started.")
    location_probs = detect_image_location(image_url, location_choices)
    assert isinstance(location_probs, dict), "Test case [1/1] failed: The function should return a dictionary."
    assert all(location in location_choices for location in location_probs.keys()), "Test case [1/1] failed: The dictionary keys should match the location choices."
    print("Testing case [1/1] completed successfully.")
    print("Testing finished.")

# Run the test function
test_detect_image_location()