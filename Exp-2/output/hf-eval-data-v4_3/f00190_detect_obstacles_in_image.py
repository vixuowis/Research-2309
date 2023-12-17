# requirements_file --------------------

import subprocess

requirements = ["transformers", "Pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_obstacles_in_image(image_path):
    """
    Detects obstacles in the image at the specified path using YOLOS object detection.

    Args:
        image_path (str): The path to the image file to be analyzed.

    Returns:
        tuple: A tuple containing the logits and bounding boxes of the detections.

    Raises:
        FileNotFoundError: If the image file does not exist at the given path.
        Exception: If there is an issue with model loading or image processing.
    """
    # Load the image
    try:
        image = Image.open(image_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found at {image_path}.") from e

    # Load the feature extractor and model
    try:
        feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
    except Exception as e:
        raise Exception("Failed to load the YOLOS model or feature extractor.") from e

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform object detection
    outputs = model(**inputs)

    # Extract the logits and bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    return logits, bboxes

# test_function_code --------------------

from os.path import exists

def test_detect_obstacles_in_image():
    print("Testing started.")
    sample_image_path = 'test_image.jpg'  # Replace with a valid image path for testing.
    
    # Test case 1: Valid image file
    print("Testing case [1/2] started.")
    if not exists(sample_image_path):
        raise FileNotFoundError("Test image does not exist.")
    logits, bboxes = detect_obstacles_in_image(sample_image_path)
    assert logits is not None and bboxes is not None, f"Test case [1/2] failed: No output received."

    # Test case 2: Non-existent image file
    print("Testing case [2/2] started.")
    non_existent_path = 'non_existent_image.jpg'
    try:
        detect_obstacles_in_image(non_existent_path)
        assert False, f"Test case [2/2] failed: No exception raised for missing file."
    except FileNotFoundError:
        pass  # Passed test as exception was correctly raised

    print("Testing finished.")

test_detect_obstacles_in_image()

# call_test_function_line --------------------

test_detect_obstacles_in_image()