# requirements_file --------------------

!pip install -U transformers pillow requests

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel


# function_code --------------------

def predict_image_geolocation(image_url, city_list):
    """Geolocalize an image by getting the probabilities of different cities.

    Args:
        image_url (str): URL of the image to geolocalize.
        city_list (list): List of city names to be considered in geolocalization.

    Returns:
        dict: A dictionary of cities and their corresponding probabilities.

    Raises:
        ValueError: If the image_url is not accessible.
        RuntimeError: If the model or processor loading fails.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except requests.RequestException:
        raise ValueError(f'Image URL {image_url} is not accessible.')
    except IOError:
        raise ValueError(f'Failed to open image from the URL {image_url}.')

    inputs = processor(text=city_list, images=image, return_tensors='pt', padding=True)
    try:
        outputs = model(**inputs)
    except RuntimeError:
        raise RuntimeError('Model or processor loading failed.')

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]
    return dict(zip(city_list, probs))


# test_function_code --------------------

def test_predict_image_geolocation():
    print("Testing started.")
    test_image_url = 'https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg'
    test_city_list = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']

    # Test case 1: URL is accessible and model prediction is successful
    print("Testing case [1/3] started.")
    try:
        results = predict_image_geolocation(test_image_url, test_city_list)
        assert isinstance(results, dict), "Test case [1/3] failed: The result should be a dictionary."
        assert len(results) == len(test_city_list), "Test case [1/3] failed: The number of results should match the number of cities."
    except Exception as e:
        assert False, f"Test case [1/3] failed: {str(e)}"

    # Test case 2: Invalid URL
    print("Testing case [2/3] started.")
    invalid_url = 'https://this_is_an_invalid_url.com/image.jpg'
    try:
        predict_image_geolocation(invalid_url, test_city_list)
        assert False, "Test case [2/3] failed: ValueError for inaccessible URL was expected."
    except ValueError:
        pass

    # Test case 3: Empty city list
    print("Testing case [3/3] started.")
    try:
        results = predict_image_geolocation(test_image_url, [])
        assert results == {}, "Test case [3/3] failed: Expected an empty dictionary for an empty city list."
    except Exception as e:
        assert False, f"Test case [3/3] failed: {str(e)}"

    print("Testing finished.")


# call_test_function_line --------------------

test_predict_image_geolocation()