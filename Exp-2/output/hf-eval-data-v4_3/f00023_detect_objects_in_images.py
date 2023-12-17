# requirements_file --------------------

import subprocess

requirements = ["requests", "Pillow", "torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests
import torch

# function_code --------------------

def detect_objects_in_images(url, texts):
    """
    Detects objects in an image based on the given text queries.

    Args:
        url (str): The URL of the image to be analyzed.
        texts (list[str]): A list of text queries that describe the objects to be detected.

    Returns:
        dict: A dictionary with the results of object detection, including the bounding boxes and labels.

    Raises:
        ValueError: If the URL or texts are not valid.
    """
    if not url or not texts:
        raise ValueError('URL and texts must be provided.')

    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=[texts], images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results

# test_function_code --------------------

def test_detect_objects_in_images():
    print("Testing started.")
    
    # Test case 1: A valid image URL and valid texts
    print("Testing case [1/3] started.")
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a cat', 'a photo of a dog']
    result = detect_objects_in_images(url, texts)
    assert 'labels' in result, "Test case [1/3] failed: 'labels' not in result"

    # Test case 2: Invalid URL
    print("Testing case [2/3] started.")
    url = ''  # Empty URL
    try:
        detect_objects_in_images(url, texts)
        assert False, "Test case [2/3] failed: ValueError was not raised for an empty URL"
    except ValueError:
        pass

    # Test case 3: Invalid texts
    print("Testing case [3/3] started.")
    texts = []  # Empty texts list
    try:
        detect_objects_in_images(url, texts)
        assert False, "Test case [3/3] failed: ValueError was not raised for empty texts list"
    except ValueError:
        pass
    
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_images()