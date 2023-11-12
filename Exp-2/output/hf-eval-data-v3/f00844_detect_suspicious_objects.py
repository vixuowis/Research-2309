# function_import --------------------

from PIL import Image
import requests
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_suspicious_objects(image_url: str, texts: list) -> dict:
    '''
    Detect suspicious objects and people in an image using a zero-shot text-conditioned object detection system.

    Args:
        image_url (str): The URL of the image to analyze.
        texts (list): A list of text descriptions that represent suspicious objects and people.

    Returns:
        dict: The detection results.
    '''
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results

# test_function_code --------------------

def test_detect_suspicious_objects():
    '''
    Test the detect_suspicious_objects function.
    '''
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a suspicious person', 'a photo of a suspicious object']
    results = detect_suspicious_objects(image_url, texts)
    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'scores' in results, 'The result should contain scores.'
    assert 'labels' in results, 'The result should contain labels.'
    assert 'boxes' in results, 'The result should contain boxes.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_suspicious_objects()