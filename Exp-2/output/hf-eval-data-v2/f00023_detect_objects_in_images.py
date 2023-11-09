# function_import --------------------

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests
import torch

# function_code --------------------

def detect_objects_in_images(url, texts):
    '''
    Detect objects in images using the OwlViTForObjectDetection model.

    Args:
        url (str): The URL of the image.
        texts (list): A list of texts representing the objects of interest.

    Returns:
        A dictionary containing the detected objects and their locations.
    '''
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch16')

    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=texts, images=image, return_tensors='pt')

    outputs = model(**inputs)
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    return results

# test_function_code --------------------

def test_detect_objects_in_images():
    '''
    Test the detect_objects_in_images function.
    '''
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = [['a photo of a living room', 'a photo of a kitchen', 'a photo of a bedroom', 'a photo of a bathroom']]
    results = detect_objects_in_images(url, texts)

    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'predictions' in results, 'The result should contain predictions.'

# call_test_function_code --------------------

test_detect_objects_in_images()