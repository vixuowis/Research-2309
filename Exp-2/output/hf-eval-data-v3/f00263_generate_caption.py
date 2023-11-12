# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def generate_caption(image_url: str) -> str:
    """
    Generate a caption for an image using the 'salesforce/blip2-opt-6.7b' model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to generate a caption for.

    Returns:
        str: The generated caption.
    """
    # Load the model
    caption_generator = pipeline('text2text-generation', model='salesforce/blip2-opt-6.7b')

    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Generate the caption
    caption = caption_generator(image)

    return caption

# test_function_code --------------------

def test_generate_caption():
    """
    Test the generate_caption function.
    """
    # Test with a cat image
    cat_caption = generate_caption('https://placekitten.com/200/300')
    assert isinstance(cat_caption, str), 'Caption should be a string'

    # Test with a dog image
    dog_caption = generate_caption('https://placedog.net/500')
    assert isinstance(dog_caption, str), 'Caption should be a string'

    # Test with a landscape image
    landscape_caption = generate_caption('https://placeimg.com/640/480/nature')
    assert isinstance(landscape_caption, str), 'Caption should be a string'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_caption()