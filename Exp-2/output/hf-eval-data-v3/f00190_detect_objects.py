# function_import --------------------

from transformers import YolosForObjectDetection
from PIL import Image

# function_code --------------------

def detect_objects(image_path):
    """
    Detect objects in an image using the YOLOS model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The model outputs including logits and predicted boxes.
    """
    image = Image.open(image_path)
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
    inputs = model.feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    return {'logits': outputs.logits, 'boxes': outputs.pred_boxes}

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    # Test with a sample image
    outputs = detect_objects('sample_image.jpg')
    assert isinstance(outputs, dict)
    assert 'logits' in outputs
    assert 'boxes' in outputs
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_objects()