# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_cat_in_image(image_url):
    """
    Detects if a cat is present in the image at the given URL using the YOLOS model.

    Args:
        image_url (str): URL of the image to be analyzed.

    Returns:
        bool: True if a cat is detected, False otherwise.

    Raises:
        ValueError: If the image_url is invalid or cannot be opened.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError("Invalid image URL or cannot open image.") from e

    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes

    cat_detected = any(['cat' in cls for cls in logits.indices])
    return cat_detected

# test_function_code --------------------

def test_detect_cat_in_image():
    print("Testing started.")
    sample_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Testing case 1: Image with a cat
    print("Testing case [1/3] started.")
    result1 = detect_cat_in_image(sample_image_url)
    assert result1 == True, f"Test case [1/3] failed: Expected True, got {result1}"

    # Testing case 2: Image without a cat (replace with appropriate URL)
    print("Testing case [2/3] started.")
    sample_image_url2 = 'http://images.cocodataset.org/val2017/000000039770.jpg'
    result2 = detect_cat_in_image(sample_image_url2)
    assert result2 == False, f"Test case [2/3] failed: Expected False, got {result2}"

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_cat_in_image()