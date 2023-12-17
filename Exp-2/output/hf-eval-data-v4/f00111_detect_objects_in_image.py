# requirements_file --------------------

!pip install -U transformers, PIL, requests

# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(image_url):
    """
    Detect objects in an image using a pretrained YOLOS Tiny model.

    Parameters:
        image_url (str): URL of the image for object detection.

    Returns:
        tuple: Contains logits and bounding boxes of the detected objects.
    """
    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Initialize the feature extractor and model
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Obtain object detection outputs
    outputs = model(**inputs)

    # Return the logits and bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    return logits, bboxes

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")

    # Test case 1: Check if the function returns a tuple
    print("Testing case [1/1] started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = detect_objects_in_image(image_url)
    assert isinstance(result, tuple), "Test case [1/1] failed: The function should return a tuple."
    print("Test case [1/1] passed.")
    print("Testing finished.")