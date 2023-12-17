# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def recommend_location(image_url, city_choices):
    """
    Recommend the most suitable city for a new store location.

    Args:
    image_url (str): The URL of the image from the potential store location.
    city_choices (list): A list of cities to be considered for the new store.

    Returns:
    list: A list of tuples with each city and its corresponding probability.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=city_choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).tolist()[0]
    return list(zip(city_choices, probs))

# test_function_code --------------------

def test_recommend_location():
    print("Testing started.")
    image_url = 'https://example.com/sample_location_image.jpg'
    choices = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']

    # Test case 1
    print("Testing case [1/1] started.")
    result = recommend_location(image_url, choices)
    assert len(result) == len(choices), f"Test case [1/1] failed: Expected {len(choices)} results, got {len(result)}"
    assert all(isinstance(item, tuple) for item in result), "Test case [1/1] failed: Expected list of tuples"
    assert all(isinstance(name, str) and isinstance(prob, float) for name, prob in result), "Test case [1/1] failed: Expected tuples with (str, float)"
    print("Testing case [1/1] completed.")
    print("Testing finished.")

# Running the test function
test_recommend_location()