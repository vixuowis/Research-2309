# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def image_to_text(image_url: str) -> str:
    """
    Generate a text description based on the content of the image.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        str: The generated text description of the image.

    Raises:
        Exception: If the image cannot be loaded from the provided URL.
    """
    # Create a text generation pipeline with the pre-trained model
    img2text_pipeline = pipeline('text-generation', model='microsoft/git-large-r-textcaps')

    # Load the image data from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Provide the image to the pipeline
    text_output = img2text_pipeline(image)[0]['generated_text']

    return text_output

# test_function_code --------------------

def test_image_to_text():
    """
    Test the image_to_text function with different image URLs.
    """
    # Test with a cat image
    cat_image_url = 'https://placekitten.com/200/300'
    cat_text = image_to_text(cat_image_url)
    assert isinstance(cat_text, str), 'The output should be a string'

    # Test with a dog image
    dog_image_url = 'https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg'
    dog_text = image_to_text(dog_image_url)
    assert isinstance(dog_text, str), 'The output should be a string'

    # Test with a landscape image
    landscape_image_url = 'https://upload.wikimedia.org/wikipedia/commons/5/5c/Natural_bridge_in_Arches_National_Park_UT.jpg'
    landscape_text = image_to_text(landscape_image_url)
    assert isinstance(landscape_text, str), 'The output should be a string'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_image_to_text()