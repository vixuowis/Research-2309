# function_import --------------------

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_url):
    """
    Detect objects in an image from a URL using the pre-trained model 'facebook/detr-resnet-101'.

    Args:
        image_url (str): The URL of the image.

    Returns:
        dict: The detected objects and their confidence scores.

    Raises:
        ImportError: If the required libraries are not installed.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_detect_objects():
    """
    Test the 'detect_objects' function.
    """
    url1 = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    url2 = 'https://placekitten.com/200/300'
    url3 = 'https://placekitten.com/400/600'
    assert isinstance(detect_objects(url1), dict)
    assert isinstance(detect_objects(url2), dict)
    assert isinstance(detect_objects(url3), dict)
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_objects()