# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def generate_image_description(img_url):
    '''
    Generate a description of an image using the BlipForConditionalGeneration model.

    Args:
        img_url (str): The URL or local path of the image to be described.

    Returns:
        str: The generated description of the image.
    '''
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, return_tensors='pt')
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# test_function_code --------------------

def test_generate_image_description():
    '''
    Test the function generate_image_description.
    '''
    img_url1 = 'https://placekitten.com/200/300'
    img_url2 = 'https://placekitten.com/400/500'
    img_url3 = 'https://placekitten.com/600/700'
    assert isinstance(generate_image_description(img_url1), str)
    assert isinstance(generate_image_description(img_url2), str)
    assert isinstance(generate_image_description(img_url3), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_description()