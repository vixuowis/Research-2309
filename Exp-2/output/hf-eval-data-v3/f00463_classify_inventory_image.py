# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image

# function_code --------------------

def classify_inventory_image(image_path):
    """
    Classify the type of an image for an inventory using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted label for the image.

    Raises:
        UnidentifiedImageError: If the image file cannot be identified.
    """
    image = Image.open(image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_inventory_image():
    """
    Test the classify_inventory_image function.
    """
    test_image_path = 'https://placekitten.com/200/300'
    result = classify_inventory_image(test_image_path)
    assert isinstance(result, str), 'The result should be a string.'
    assert result != '', 'The result should not be an empty string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_inventory_image()