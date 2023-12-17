# requirements_file --------------------

import subprocess

requirements = ["transformers", "PIL", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(url):
    """
    Detect objects in an image using a pretrained YOLOS Tiny model.

    Args:
        url (str): URL of the image to process.

    Returns:
        dict: A dictionary containing the logits and bounding boxes of detected objects.

    Raises:
        ValueError: If the image cannot be loaded from the URL.
    """
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Image cannot be loaded from URL. Error: {e}')

    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return {
        'logits': outputs.logits,
        'bboxes': outputs.pred_boxes
    }

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    print("Testing case [1/1] started.")
    results = detect_objects_in_image(test_url)
    assert 'logits' in results and 'bboxes' in results, f"Test case [1/1] failed: Expected 'logits' and 'bboxes' in results."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_objects_in_image()