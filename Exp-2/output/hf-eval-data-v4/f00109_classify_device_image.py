# requirements_file --------------------

!pip install -U transformers Pillow

# function_import --------------------

from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# function_code --------------------

def classify_device_image(image_path):
    """
    Classify an image of a device as a cell phone, laptop, or smartwatch.

    Parameters:
    image_path (str): The file path or URL to the image to be classified.

    Returns:
    int: The index of the device type where 0 corresponds to a cell phone,
         1 to a laptop, and 2 to a smartwatch.
    """
    model = ViTForImageClassification.from_pretrained('lysandre/tiny-vit-random')
    feature_extractor = ViTFeatureExtractor.from_pretrained('lysandre/tiny-vit-random')
    image = Image.open(image_path)
    input_image = feature_extractor(images=image, return_tensors='pt')
    output = model(**input_image)
    device_type = output.logits.argmax(dim=1).item()

    return device_type

# test_function_code --------------------

def test_classify_device_image():
    print("Testing classify_device_image() function.")
    assert classify_device_image('cell_phone_sample.jpg') == 0, "Test failed: 'cell_phone_sample.jpg' should be classified as a cell phone."
    assert classify_device_image('laptop_sample.jpg') == 1, "Test failed: 'laptop_sample.jpg' should be classified as a laptop."
    assert classify_device_image('smartwatch_sample.jpg') == 2, "Test failed: 'smartwatch_sample.jpg' should be classified as a smartwatch."
    print("All tests passed.")