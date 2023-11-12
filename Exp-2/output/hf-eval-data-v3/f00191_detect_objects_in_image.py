# function_import --------------------

from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(image_url: str):
    """
    Detect objects in an image using the DeformableDetrForObjectDetection model.

    Args:
        image_url (str): The URL of the image to process.

    Returns:
        dict: The outputs of the model.

    Raises:
        ImportError: If the required libraries are not installed.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_detect_objects_in_image():
    """
    Test the detect_objects_in_image function.
    """
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    outputs = detect_objects_in_image(sample_image_url)
    assert isinstance(outputs, dict), 'The output should be a dictionary.'
    assert 'pred_logits' in outputs, 'The output dictionary should have a key named "pred_logits".'
    assert 'pred_boxes' in outputs, 'The output dictionary should have a key named "pred_boxes".'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_objects_in_image()