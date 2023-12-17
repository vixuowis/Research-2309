# requirements_file --------------------

!pip install -U transformers requests Pillow

# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_objects_in_image(image_url, text_queries):
    """
    Identify objects within an image based on textual descriptions.

    Parameters:
        image_url (str): URL of the image to process.
        text_queries (list): List of textual descriptions to identify in the image.

    Returns:
        dict: object detection information for specified text queries in the image.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')
    
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=text_queries, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])

    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing detect_objects_in_image function.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    test_queries = ['a photo of a dog']

    # Test case: Detecting objects
    print("Testing object detection.")
    results = detect_objects_in_image(test_url, test_queries)
    assert len(results) > 0, "Object detection failed: no results returned."
    print("Object detection test passed.")

    # Further test cases can be added as needed.

    print("All tests passed!")

test_detect_objects_in_image()