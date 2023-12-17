# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch

# function_code --------------------

def classify_security_camera_image(image):
    """
    Classifies an image captured by a security camera using a pretrained RegNet model.

    Args:
        image (str): An image file path or image as BytesIO object.

    Returns:
        str: The label of the image as classified by the model.

    Raises:
        ValueError: If the image is not in the correct format.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_security_camera_image():
    print("Testing started.")
    # Simulate an image
    sample_image = torch.zeros((3, 224, 224))

    # Test case 1: Verify the function returns a string
    print("Testing case [1/1] started.")
    label = classify_security_camera_image(sample_image)
    assert isinstance(label, str), f"Test case [1/1] failed: Expected str, got {type(label)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_security_camera_image()