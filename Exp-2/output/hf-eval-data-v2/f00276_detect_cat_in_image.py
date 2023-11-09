# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
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
    """
    image = Image.open(requests.get(image_url, stream=True).raw)

    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Check if 'cat' class is present in the predicted object classes
    cat_detected = any([cls == 'cat' for cls in logits.indices])

    return cat_detected

# test_function_code --------------------

def test_detect_cat_in_image():
    """
    Tests the 'detect_cat_in_image' function with a sample image URL.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    assert isinstance(detect_cat_in_image(image_url), bool)

# call_test_function_code --------------------

test_detect_cat_in_image()