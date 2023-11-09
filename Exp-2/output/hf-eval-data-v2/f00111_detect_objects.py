# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_url):
    """
    Detects objects in an image using the YOLOS Tiny model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to detect objects in.

    Returns:
        logits (torch.Tensor): The classification scores for each object detected in the image.
        bboxes (torch.Tensor): The predicted bounding boxes for each object detected in the image.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    return logits, bboxes

# test_function_code --------------------

def test_detect_objects():
    """
    Tests the detect_objects function by detecting objects in a sample image.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    logits, bboxes = detect_objects(image_url)
    assert logits is not None
    assert bboxes is not None

# call_test_function_code --------------------

test_detect_objects()