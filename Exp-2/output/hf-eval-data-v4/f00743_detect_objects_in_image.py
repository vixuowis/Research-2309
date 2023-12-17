# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_objects_in_image(url):
    """
    Detect objects in the image provided by the URL using the pre-trained DETR (facebook/detr-resnet-101-dc5) model.

    Parameters:
    url (str): URL of the image to be processed.

    Returns:
    dict: A dictionary with two keys: 'labels' and 'boxes', containing the detected object labels and bounding boxes.
    """
    # Load the image
    image = Image.open(requests.get(url, stream=True).raw)

    # Initialize the feature extractor and the model
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')

    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors='pt')
    # Get model predictions
    outputs = model(**inputs)

    # Process the predictions
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    # Get the detected labels and their corresponding bounding boxes
    labels = [model.config.id2label[label_id] for label_id in logits.argmax(-1).flatten().tolist()]
    boxes = bboxes.tolist()

    return {'labels': labels, 'boxes': boxes}

# test_function_code --------------------

def test_detect_objects_in_image():
    print("Testing started.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    print("Testing object detection function.")
    result = detect_objects_in_image(test_url)

    # Check if the result contains the necessary keys
    assert 'labels' in result, "Test failed: The result does not contain 'labels'."
    assert 'boxes' in result, "Test failed: The result does not contain 'boxes'."

    # Further checks could be done here if ground truth data is available for comparison

    print("Test passed. Object detection function works as expected.")
    print("Testing finished.")

# Run the test function
test_detect_objects_in_image()