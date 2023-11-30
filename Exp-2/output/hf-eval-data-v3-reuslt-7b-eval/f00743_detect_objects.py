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

    # Load the pretrained model from Hugging Face Transformers. The first time this is run, it may take a few minutes to download the model.
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    # Load the image from the specified URL.  If this is not a local file, you can remove the "open" step and just pass in the URL to Image.open().
    with open(requests.get(image_url, stream=True).raw, 'rb') as f:
        image = Image.open(f)
        
    # Transform the image into a format that DetrFeatureExtractor expects.
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Run the model forward to make predictions about the image.
    outputs = model(**inputs)
    
    # Extract and return the logits and bounding boxes from the output.
    return outputs.logits, outputs.pred_boxes[0]

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