# requirements_file --------------------

import subprocess

requirements = ["pillow", "requests", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def get_location_recommendation(image_url, city_names):
    """
    This function uses the StreetCLIP model to recommend the most suitable city for a new store
    based on an image from a potential location.

    Args:
        image_url (str): The URL of the image from the potential location.
        city_names (List[str]): A list of city names to generate probabilities for.

    Returns:
        dict: A dictionary containing each city name and its associated probability.

    Raises:
        ValueError: If the image URL is invalid or if the image cannot be loaded.
    """
    # Load the pretrained StreetCLIP model
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')

    # Try to load the image from the URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Failed to load image from the provided URL: {e}')

    # Process the text and image input
    inputs = processor(text=city_names, images=image, return_tensors='pt', padding=True)

    # Get model output logits and compute probabilities
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Create a dictionary of city names and their probabilities
    recommendations = {city: float(prob) for city, prob in zip(city_names, probs.squeeze().tolist())}

    return recommendations

# test_function_code --------------------

def test_get_location_recommendation():
    print("Testing started.")
    # Test with a sample image and city names
    image_url = 'https://example.com/sample_image.jpg'
    city_names = ['San Francisco', 'New York', 'Tokyo']

    print("Testing case [1/1] started.")
    recommendations = get_location_recommendation(image_url, city_names)
    assert isinstance(recommendations, dict), f"Test case failed: Expected a dictionary, got {type(recommendations)}"
    assert all(city in recommendations for city in city_names), f"Test case failed: Not all cities found in recommendations"
    print("Testing finished.")

# call_test_function_line --------------------

test_get_location_recommendation()