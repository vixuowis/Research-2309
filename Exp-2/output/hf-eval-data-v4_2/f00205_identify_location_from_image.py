# requirements_file --------------------

!pip install -U pillow requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def identify_location_from_image(url, possible_locations):
    """
    Identifies the most likely geographic location of an image using a pretrained geolocalization model.

    Args:
        url (str): URL of the image to be analyzed.
        possible_locations (List[str]): A list of possible locations to compare against.

    Returns:
        Tuple(float, str): A tuple containing the highest probability and the corresponding location.

    Raises:
        ValueError: If `url` is not a valid URL or `possible_locations` is empty.
    """
    # Load the pretrained models
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')

    # Retrieve and process the image
    image = Image.open(requests.get(url, stream=True).raw)

    # Check if the list of possible locations is empty
    if not possible_locations:
        raise ValueError('The list of possible locations cannot be empty.')

    # Prepare inputs
    inputs = processor(text=possible_locations, images=image, return_tensors='pt', padding=True)

    # Get model outputs
    outputs = model(**inputs)

    # Calculate probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_prob, max_idx = probs.max(dim=1)

    # Return the highest probability and its corresponding location
    return max_prob.item(), possible_locations[max_idx.item()]

# test_function_code --------------------

def test_identify_location_from_image():
    print('Testing started.')

    # Set up test data
    test_url = 'https://image_url_here.jpeg'
    test_locations = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    # Test case 1: Valid url and non-empty locations list
    print('Testing case [1/3] started.')
    prob, location = identify_location_from_image(test_url, test_locations)
    assert prob >= 0.0 and prob <= 1.0, f'Test case [1/3] failed: Probability out of bounds'
    assert location in test_locations, f'Test case [1/3] failed: Invalid location'

    # Test case 2: Invalid image url
    print('Testing case [2/3] started.')
    invalid_url = 'not_a_valid_url'
    try:
        identify_location_from_image(invalid_url, test_locations)
        assert False, 'Test case [2/3] failed: ValueError not raised for invalid url'
    except ValueError:
        pass

    # Test case 3: Empty locations list
    print('Testing case [3/3] started.')
    empty_locations = []
    try:
        identify_location_from_image(test_url, empty_locations)
        assert False, 'Test case [3/3] failed: ValueError not raised for empty locations list'
    except ValueError:
        pass

    print('Testing finished.')

# call_test_function_line --------------------

test_identify_location_from_image()