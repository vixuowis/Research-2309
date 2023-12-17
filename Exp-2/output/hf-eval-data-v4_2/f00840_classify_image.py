# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image using a Swin Transformer model.

    Args:
        image_url (str): URL of the image to be classified.

    Returns:
        str: The predicted class label of the image.

    Raises:
        ValueError: If image_url is not accessible.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except requests.exceptions.RequestException as e:
        raise ValueError(f'Unable to access image URL: {image_url}') from e

    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Test case 1: Known image URL
    print("Testing case [1/2] started.")
    predicted_label = classify_image(test_url)
    assert predicted_label, f"Test case [1/2] failed: Expected a class label, got {predicted_label}"

    # Test case 2: Invalid image URL
    print("Testing case [2/2] started.")
    try:
        classify_image('http://invalid_url.jpg')
        assert False, "Test case [2/2] failed: Expected a ValueError for invalid URL"
    except ValueError as e:
        assert str(e), f"Test case [2/2] failed: Expected ValueError with a message, got {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()