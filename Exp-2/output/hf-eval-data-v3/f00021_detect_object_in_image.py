# function_import --------------------

import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# function_code --------------------

def detect_object_in_image(url: str, texts: list) -> dict:
    """
    Identify an object within an image based on textual description.

    Args:
        url (str): The URL of the image.
        texts (list): The list of text queries to identify objects in the image.

    Returns:
        dict: The object detection results.

    Raises:
        requests.exceptions.RequestException: If the image cannot be downloaded from the URL.
        RuntimeError: If the model cannot be loaded.
    """
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')

    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except requests.exceptions.RequestException as e:
        raise RuntimeError('Failed to download image from URL.') from e

    inputs = processor(text=texts, images=image, return_tensors='pt')
    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])

    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    return results

# test_function_code --------------------

def test_detect_object_in_image():
    """Test the detect_object_in_image function."""
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a dog']
    results = detect_object_in_image(url, texts)
    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'scores' in results, 'The result should contain scores.'
    assert 'labels' in results, 'The result should contain labels.'
    assert 'boxes' in results, 'The result should contain boxes.'

    url = 'https://placekitten.com/200/300'
    texts = ['a photo of a cat']
    results = detect_object_in_image(url, texts)
    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'scores' in results, 'The result should contain scores.'
    assert 'labels' in results, 'The result should contain labels.'
    assert 'boxes' in results, 'The result should contain boxes.'

    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_detect_object_in_image()