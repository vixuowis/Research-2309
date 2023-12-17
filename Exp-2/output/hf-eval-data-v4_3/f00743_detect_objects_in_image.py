# requirements_file --------------------

import subprocess

requirements = ["transformers", "Pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(image_url):
    """
    Detect objects in the image located at the given URL using a pre-trained DETR model.

    Args:
        image_url (str): The URL of the image to process.

    Returns:
        dict: A dictionary containing the logits and bounding boxes of the detections.

    Raises:
        ValueError: If the image at the provided URL cannot be opened.
    """
    # Load the pre-trained model and feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    
    # Download and open the image
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image = Image.open(response.raw)
    else:
        raise ValueError(f'Image cannot be opened, URL may be invalid or inaccessible. Status code: {response.status_code}')
    
    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # Get predictions
    outputs = model(**inputs)
    
    # Extract logits and bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    
    return {
        'logits': logits,
        'bounding_boxes': bboxes
    }

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")
    # Test with a valid image URL
    print("Testing case [1/2] started.")
    result = detect_objects_in_image('http://images.cocodataset.org/val2017/000000039769.jpg')
    assert 'logits' in result and 'bounding_boxes' in result, "Test case [1/2] failed: 'logits' or 'bounding_boxes' not in result"
    
    # Test with an invalid image URL
    print("Testing case [2/2] started.")
    try:
        detect_objects_in_image('http://invalid_url')
        assert False, "Test case [2/2] failed: ValueError expected but not raised."
    except ValueError as e:
        assert str(e) == 'Image cannot be opened, URL may be invalid or inaccessible. Status code: 404', f"Test case [2/2] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_image()