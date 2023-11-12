# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def get_city_probabilities(image_url: str, city_choices: list) -> dict:
    '''
    Get the probabilities of different cities for the given image.

    Args:
        image_url (str): The URL of the image to be geolocalized.
        city_choices (list): A list of city names to geolocalize the image to.

    Returns:
        dict: A dictionary with city names as keys and their corresponding probabilities as values.

    Raises:
        Exception: If there is an error in fetching the image or processing the image and texts.
    '''
    try:
        model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
        processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(text=city_choices, images=image, return_tensors='pt', padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()[0]
        city_probs = dict(zip(city_choices, probs))
        return city_probs
    except Exception as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_get_city_probabilities():
    '''
    Test the function get_city_probabilities.
    '''
    image_url = 'https://placekitten.com/200/300'
    city_choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']
    city_probs = get_city_probabilities(image_url, city_choices)
    assert isinstance(city_probs, dict), 'The result should be a dictionary.'
    assert len(city_probs) == len(city_choices), 'The number of cities should be equal to the number of choices.'
    assert sum(city_probs.values()) == 1, 'The sum of probabilities should be 1.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_get_city_probabilities()