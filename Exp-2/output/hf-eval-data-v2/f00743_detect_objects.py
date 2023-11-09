# function_import --------------------

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_url):
    """
    Detect objects in an image using the pre-trained DETR model.

    Args:
        image_url (str): URL of the image to be processed.

    Returns:
        logits (torch.Tensor): The classification logits (including no-object) for all queries.
        bboxes (torch.Tensor): The predicted bounding boxes in [x0, y0, x1, y1] format.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.logits, outputs.pred_boxes

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function with a sample image from the COCO 2017 validation dataset.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    logits, bboxes = detect_objects(image_url)
    assert logits is not None
    assert bboxes is not None
    assert logits.size(0) == bboxes.size(0)

# call_test_function_code --------------------

test_detect_objects()