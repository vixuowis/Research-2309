# function_import --------------------

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects(url):
    '''
    Detect objects in an image from a URL using the pre-trained model 'facebook/detr-resnet-101'.
    
    Args:
        url (str): The URL of the image.
    
    Returns:
        dict: The detected objects and their confidence scores.
    
    Raises:
        Exception: If the image cannot be opened.
    '''
    image = Image.open(requests.get(url, stream=True).raw)
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_detect_objects():
    '''
    Test the 'detect_objects' function.
    
    Raises:
        AssertionError: If the function does not work as expected.
    '''
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    outputs = detect_objects(url)
    assert isinstance(outputs, dict), 'The output should be a dictionary.'
    assert 'pred_logits' in outputs, 'The output should contain predicted logits.'
    assert 'pred_boxes' in outputs, 'The output should contain predicted boxes.'

# call_test_function_code --------------------

test_detect_objects()