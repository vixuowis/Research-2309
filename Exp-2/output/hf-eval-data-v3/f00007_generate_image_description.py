# function_import --------------------

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def generate_image_description(image_url):
    '''
    Generate a description for an image provided as input.
    
    Args:
        image_url (str): The URL of the image to be described.
    
    Returns:
        str: The generated description of the image.
    
    Raises:
        Exception: If the image cannot be loaded from the provided URL.
    '''
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    description_generator = pipeline('text-generation', model='microsoft/git-large-r-textcaps')
    image_description = description_generator(image)
    return image_description

# test_function_code --------------------

def test_generate_image_description():
    '''
    Test the generate_image_description function.
    '''
    test_image_url_1 = 'https://placekitten.com/200/300'
    test_image_url_2 = 'https://placekitten.com/400/500'
    test_image_url_3 = 'https://placekitten.com/600/700'
    assert isinstance(generate_image_description(test_image_url_1), str)
    assert isinstance(generate_image_description(test_image_url_2), str)
    assert isinstance(generate_image_description(test_image_url_3), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_description()