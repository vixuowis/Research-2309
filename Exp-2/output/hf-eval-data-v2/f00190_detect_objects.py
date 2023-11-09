# function_import --------------------

from transformers import YolosForObjectDetection, YolosFeatureExtractor
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_path):
    """
    Detect objects in an image using the YOLOS model fine-tuned on COCO 2017 object detection.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A tuple (logits, bboxes), where logits are the classification scores for each object detected in the image,
        and bboxes are the bounding boxes of the detected objects.
    """
    # Load the image data
    image = Image.open(image_path)

    # Load the pre-trained model
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    # Load the feature extractor
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Run the model
    outputs = model(**inputs)

    return outputs.logits, outputs.pred_boxes

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    # Define the image URL
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Download the image
    image_path = 'test_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(requests.get(url).content)

    # Run the function
    logits, bboxes = detect_objects(image_path)

    # Check the outputs
    assert logits is not None, 'No logits were returned'
    assert bboxes is not None, 'No bounding boxes were returned'

    # Clean up
    os.remove(image_path)

# call_test_function_code --------------------

test_detect_objects()