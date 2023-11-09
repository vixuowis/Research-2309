# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_object_in_image(url: str, texts: list):
    """
    Identify an object within an image based on textual description using OwlViTForObjectDetection model.

    Args:
        url (str): The URL of the image.
        texts (list): The list of text queries to identify objects in the image.

    Returns:
        dict: The post-processed results of object detection.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')

    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])

    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results

# test_function_code --------------------

def test_detect_object_in_image():
    """
    Test the function detect_object_in_image.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a dog']
    results = detect_object_in_image(url, texts)

    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'predictions' in results, 'The key "predictions" should be in the results.'

# call_test_function_code --------------------

test_detect_object_in_image()