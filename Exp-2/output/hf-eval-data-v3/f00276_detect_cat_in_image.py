# function_import --------------------

from transformers import YolosForObjectDetection, YolosFeatureExtractor
from PIL import Image
import requests

# function_code --------------------

def detect_cat_in_image(image_url):
    """
    Detects if there is a cat in the given image.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        bool: True if a cat is detected in the image, False otherwise.

    Raises:
        Exception: If there is an error in processing the image or in the object detection.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    cat_detected = any([cls == 'cat' for cls in logits.indices])
    return cat_detected

# test_function_code --------------------

def test_detect_cat_in_image():
    """
    Tests the function 'detect_cat_in_image'.
    """
    assert detect_cat_in_image('https://placekitten.com/200/300') == True
    assert detect_cat_in_image('https://placekitten.com/g/200/300') == False
    assert detect_cat_in_image('https://placekitten.com/200/287') == True
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_cat_in_image()