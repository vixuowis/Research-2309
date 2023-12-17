# requirements_file --------------------

import subprocess

requirements = ["requests", "pillow", "torch", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_object_in_image(image_url, text_queries):
    """
    Detects the presence of objects in the image based on the provided text queries.

    Args:
        image_url (str): The URL of the image in which to detect objects.
        text_queries (List[str]): A list of text queries describing the objects to detect.

    Returns:
        Dict[str, Any]: A dictionary with object detection information.

    Raises:
        ValueError: If the image cannot be loaded from the provided URL.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')

    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image = Image.open(response.raw)
    else:
        raise ValueError('Could not load image from the URL.')

    inputs = processor(text=text_queries, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results

# test_function_code --------------------

def test_detect_object_in_image():
    print("Testing started.")
    # Test case 1: Valid image URL and text query
    print("Testing case [1/1] started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    text_queries = ['a photo of a dog']
    try:
        results = detect_object_in_image(image_url, text_queries)
        assert isinstance(results, dict), 'Results should be a dictionary.'
    except ValueError as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_object_in_image()