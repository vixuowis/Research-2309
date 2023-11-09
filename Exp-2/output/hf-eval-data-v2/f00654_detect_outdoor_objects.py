# function_import --------------------

from transformers import OwlViTForObjectDetection, OwlViTProcessor
from PIL import Image
import requests
import torch

# function_code --------------------

def detect_outdoor_objects(image_url):
    """
    Detects outdoor objects in an image using the OwlViTForObjectDetection model.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        A dictionary containing the detected objects and their locations.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')

    image = Image.open(requests.get(image_url, stream=True).raw)

    texts = [['a tent', 'a backpack', 'hiking boots', 'a campfire', 'a kayak']]
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results

# test_function_code --------------------

def test_detect_outdoor_objects():
    """
    Tests the detect_outdoor_objects function.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    results = detect_outdoor_objects(image_url)

    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'boxes' in results, 'The result should contain bounding boxes.'
    assert 'labels' in results, 'The result should contain labels.'
    assert 'scores' in results, 'The result should contain scores.'

# call_test_function_code --------------------

test_detect_outdoor_objects()