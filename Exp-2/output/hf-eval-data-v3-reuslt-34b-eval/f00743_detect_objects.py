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
    
    # Load the feature extractor and model from Hugging Face Model Hub
    print("Loading feature extractor and model...")
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    # Download the image from the URL using requests
    print(f"Downloading image from {image_url}...")
    r = requests.get(image_url)

    # Load the image into PIL format
    print("Loading image into PIL format...")
    image = Image.open(r.raw)
    
    print("Generating predictions...")
    outputs = model(**feature_extractor(images=image, return_tensors="pt"))
    
    # Return the logits and bounding boxes from the predictions
    print("Returning outputs...")
    return outputs.logits, outputs.pred_boxes


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