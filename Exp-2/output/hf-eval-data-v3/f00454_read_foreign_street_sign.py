# function_import --------------------

from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from PIL import Image
import requests

# function_code --------------------

def read_foreign_street_sign(image_url):
    '''
    This function reads a street sign in a foreign language using the MgpstrForSceneTextRecognition model.

    Args:
        image_url (str): The URL of the image of the street sign.

    Returns:
        str: The text recognized from the street sign image.
    '''
    processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
    model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    outputs = model(pixel_values)
    generated_text = processor.batch_decode(outputs.logits)['generated_text']
    return generated_text

# test_function_code --------------------

def test_read_foreign_street_sign():
    '''
    This function tests the read_foreign_street_sign function with a sample image URL.
    '''
    image_url = 'https://i.postimg.cc/ZKwLg2Gw/367-14.png'
    result = read_foreign_street_sign(image_url)
    assert isinstance(result, str), 'The result should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_read_foreign_street_sign()