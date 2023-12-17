# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(image_url):
    """
    Detect objects in an image using a pre-trained Deformable DETR model.

    Args:
        image_url (str): URL of the image where objects are to be detected.

    Returns:
        dict: The detected objects along with their scores and bounding boxes.

    Raises:
        ValueError: If image_url is not reachable or invalid.

    """
    # Load image from the URL
    response = requests.get(image_url, stream=True)
    try:
        image = Image.open(response.raw)
    except IOError:
        raise ValueError('Unable to load image from the provided URL.')

    # Instantiate the processor and model
    processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
    model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')

    # Process the image
    inputs = processor(images=image, return_tensors='pt')
    # Detect objects
    outputs = model(**inputs)
    # Extract results
    result_data = outputs['pred_logits'].softmax(-1)[0, :, :-1].max(-1)
    scores = result_data.values
    labels = model.config.id2label[result_data.indices.tolist()]
    boxes = outputs['pred_boxes'][0].tolist()

    # Combine results into a structured dictionary
    detected_objects = [
        {'label': label, 'score': round(score.item(), 4), 'box': box}
        for label, score, box in zip(labels, scores, boxes)
        if score > 0.9  # You can adjust the threshold as needed
    ]

    return detected_objects

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # A sample image URL from COCO dataset

    # Test case 1: Valid image URL
    print("Testing case [1/1] started.")
    detected_objects = detect_objects_in_image(image_url)
    assert isinstance(detected_objects, list), "Test case [1/1] failed: The function should return a list."
    assert all('label' in obj for obj in detected_objects), "Test case [1/1] failed: All objects must have a 'label'."
    assert all('score' in obj for obj in detected_objects), "Test case [1/1] failed: All objects must have a 'score'."
    assert all('box' in obj for obj in detected_objects), "Test case [1/1] failed: All objects must have a 'box'."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_image()