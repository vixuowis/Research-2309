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
    """
    # Load the image data from a file that represents the inventory item
    image = Image.open(image_path)

    # Use the from_pretrained method of the RegNetForImageClassification class to load the pre-trained model
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')

    # Use the feature_extractor to process the image data and the model to make a prediction on the image class
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label for the image
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_inventory_image():
    """
    Test the classify_inventory_image function.
    """
    # Use a test image
    test_image_path = 'test_image.jpg'

    # Call the function with the test image
    result = classify_inventory_image(test_image_path)

    # Check the result is a string (the label)
    assert isinstance(result, str)

# call_test_function_code --------------------

test_classify_inventory_image()