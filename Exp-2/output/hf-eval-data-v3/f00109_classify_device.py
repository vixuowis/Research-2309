# function_import --------------------

from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# function_code --------------------

def classify_device(image_path: str) -> int:
    """
    Classify an image of a cell phone, laptop, or smartwatch as one of these respective device types.

    Args:
        image_path (str): The path to the image file.

    Returns:
        int: The classified device type.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    model = ViTForImageClassification.from_pretrained('lysandre/tiny-vit-random')
    feature_extractor = ViTFeatureExtractor.from_pretrained('lysandre/tiny-vit-random')
    image = Image.open(image_path)
    input_image = feature_extractor(images=image, return_tensors='pt')
    output = model(**input_image)
    device_type = output.logits.argmax(dim=1).item()
    return device_type

# test_function_code --------------------

def test_classify_device():
    """Test the classify_device function."""
    assert classify_device('test_device_image.jpg') == 0
    assert classify_device('test_device_image2.jpg') == 1
    assert classify_device('test_device_image3.jpg') == 2
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_device()