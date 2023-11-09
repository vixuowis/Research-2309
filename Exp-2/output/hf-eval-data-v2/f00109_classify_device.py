# function_import --------------------

from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# function_code --------------------

def classify_device(image_path):
    """
    Classify an image of a device into one of the categories: cell phone, laptop, or smartwatch.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The classified device type.
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
    """
    Test the classify_device function.
    """
    image_path = 'test_device_image.jpg'
    # replace 'test_device_image.jpg' with your test image file path
    device_type = classify_device(image_path)
    assert isinstance(device_type, int), 'The output should be an integer.'

# call_test_function_code --------------------

test_classify_device()