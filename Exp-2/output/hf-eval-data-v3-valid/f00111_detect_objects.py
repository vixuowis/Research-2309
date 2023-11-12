# function_import --------------------

from transformers import YolosForObjectDetection, YolosFeatureExtractor
from PIL import Image
import requests

# function_code --------------------

def detect_objects(image_url):
    """
    Detect objects in an image using the YOLOS Tiny model.

    Args:
        image_url (str): URL of the image to be processed.

    Returns:
        dict: A dictionary containing 'logits' and 'pred_boxes' which represent the detected objects and their bounding boxes respectively.
    """
    from transformers import YolosForObjectDetection, YolosFeatureExtractor
    from PIL import Image
    import requests

    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Load the feature extractor and model
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get the outputs
    outputs = model(**inputs)

    return {'logits': outputs.logits, 'pred_boxes': outputs.pred_boxes}

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    # Define a list of image URLs for testing
    test_images = [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        'https://placekitten.com/200/300',
        'https://placekitten.com/400/600'
    ]

    for image_url in test_images:
        result = detect_objects(image_url)

        # Check that the result is a dictionary
        assert isinstance(result, dict)

        # Check that the dictionary has the correct keys
        assert 'logits' in result
        assert 'pred_boxes' in result

    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_objects()