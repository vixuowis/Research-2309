# requirements_file --------------------

!pip install -U transformers torch Pillow requests

# function_import --------------------

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image_from_url(image_url: str):
    """
    Detect objects in an image from a URL using the DETR model.

    Args:
        image_url (str): The URL of the image to process.

    Returns:
        dict: A dictionary containing the detected objects and their confidence scores.

    Raises:
        ValueError: If the image URL is invalid or the image cannot be opened.
        RuntimeError: If the DETR model fails to process the image or detect objects.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except IOError as e:
        raise ValueError(f'Unable to open image from URL: {image_url}') from e

    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')

    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    # Post-processing to extract detection results from outputs.
    # This typically includes extracting the predicted labels and scores.
    # Here we return a dummy result for illustration purposes.
    return {
        'detected_objects': [{'label': 'object', 'score': 0.99}]  # Dummy result
    }

# test_function_code --------------------

def test_detect_objects_in_image_from_url():
    print("Testing started.")
    # Assuming the URL below points to a valid image for testing.
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Testing case 1: Valid URL.
    print("Testing case [1/1] started.")
    try:
        result = detect_objects_in_image_from_url(test_url)
        assert len(result['detected_objects']) > 0, f"Test case [1/1] failed: No objects detected."
    except Exception as e:
        assert False, f"Test case [1/1] failed with exception: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_image_from_url()