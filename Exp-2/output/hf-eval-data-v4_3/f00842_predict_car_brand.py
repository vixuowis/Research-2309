# requirements_file --------------------

import subprocess

requirements = ["transformers", "PIL", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests


# function_code --------------------

def predict_car_brand(image_url):
    """
    Predicts the car brand from an image URL using a pre-trained image classification model.

    Args:
        image_url (str): The URL of the image that we want to analyze.

    Returns:
        str: Predicted car brand label.

    Raises:
        ValueError: If the image cannot be loaded.
        RuntimeError: If the prediction process fails.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Image cannot be loaded.') from e

    processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    model = AutoModelForImageClassification.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    try:
        return model.config.id2label[predicted_class_idx]
    except Exception as e:
        raise RuntimeError('Prediction process failed.') from e


# test_function_code --------------------

def test_predict_car_brand():
    print("Testing started.")
    # Replace with the actual test image URL
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Test case 1
    print("Testing case [1/1] started.")
    predicted_label = predict_car_brand(test_image_url)
    assert isinstance(predicted_label, str), f"Test case [1/1] failed: Expected a string label, got {type(predicted_label)}"
    print("Test case [1/1] succeeded.")
    print("Testing finished.")


# call_test_function_line --------------------

test_predict_car_brand()