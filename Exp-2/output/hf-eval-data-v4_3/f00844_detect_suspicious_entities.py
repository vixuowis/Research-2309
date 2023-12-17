# requirements_file --------------------

import subprocess

requirements = ["Pillow", "requests", "torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
import requests
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_suspicious_entities(image_url, query_texts):
    """
    Detects suspicious objects and people in an image using a pre-trained OwlViT model.

    Args:
        image_url (str): URL of the image in which to perform detection.
        query_texts (list of str): Text descriptions of entities to identify as suspicious.

    Returns:
        dict: Detection results including bounding boxes and class labels.

    Raises:
        ValueError: If the image URL is invalid or cannot be opened.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Failed to open image from URL.') from e

    inputs = processor(text=query_texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results

# test_function_code --------------------

def test_detect_suspicious_entities():
    print("Testing started.")
    # Test case 1: Detecting a suspicious person in an image
    print("Testing case [1/2] started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    query_texts = ['a photo of a suspicious person']
    results = detect_suspicious_entities(image_url, query_texts)
    assert type(results) is dict and 'boxes' in results,
        "Test case [1/2] failed: Detection results should be a dict with 'boxes'."

    # Test case 2: Detecting a suspicious object in an image
    print("Testing case [2/2] started.")
    query_texts = ['a photo of a suspicious object']
    results = detect_suspicious_entities(image_url, query_texts)
    assert type(results) is dict and 'boxes' in results,
        "Test case [2/2] failed: Detection results should be a dict with 'boxes'."

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_suspicious_entities()