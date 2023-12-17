# requirements_file --------------------

!pip install -U transformers torch PIL requests

# function_import --------------------

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects_from_url(image_url):
    """
    Detects objects in the image provided via URL using the DETR (DEtection TRansformer) model.

    Args:
        image_url (str): The URL of the image to process

    Returns:
        dict: A dictionary containing detected objects and additional information
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')

    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    # Extract detected objects and their related information
    result = {'objects': [], 'details': outputs}
    for i in range(outputs.logits.shape[1]):
        if outputs.logits[0][i, 4].item() > 0.9:  # Considering score threshold
            result['objects'].append(processor.int_to_label(outputs.logits[0][i].argmax()))

    return result

# test_function_code --------------------

def test_detect_objects_from_url():
    print("Testing started.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Test case 1: Check the type of result
    print("Testing case [1/1] started.")
    result = detect_objects_from_url(test_url)
    assert isinstance(result, dict), "Test case [1/1] failed: Result should be a dictionary."
    assert 'objects' in result, "Test case [1/1] failed: 'objects' key should be in result."
    assert isinstance(result['objects'], list), "Test case [1/1] failed: 'objects' should be a list."
    print("Testing finished.")

# Run the test function
test_detect_objects_from_url()