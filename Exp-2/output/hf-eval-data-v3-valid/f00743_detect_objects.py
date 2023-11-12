# function_import --------------------

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_url):
    """
    Detect objects in an image using the DetrForObjectDetection model from Hugging Face Transformers.

    Args:
        image_url (str): URL of the image to process.

    Returns:
        tuple: A tuple containing the logits and bounding boxes of detected objects.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    return logits, bboxes

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function with a few test cases.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    logits, bboxes = detect_objects(image_url)
    assert logits is not None
    assert bboxes is not None
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_objects()