# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# function_code --------------------

def detect_cat_in_image(image_url):
    """
    Detects if there is a cat in the image provided by the image URL.

    Parameters:
    - image_url (str): A publicly accessible URL of the image to analyze.

    Returns:
    - bool: True if a cat is detected, False otherwise.
    """
    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Initialize the feature extractor and model
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get the model predictions
    outputs = model(**inputs)

    # Retrieve the predicted classes (index 0 corresponds to the class ID for 'cat')
    predicted_classes = outputs.logits.argmax(-1).squeeze().tolist()

    # Check if 'cat' class ID is in the predicted classes
    return 0 in predicted_classes

# test_function_code --------------------

def test_detect_cat_in_image():
    print("Testing detect_cat_in_image function.")

    # Test case 1: Image with a cat
    cat_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    assert detect_cat_in_image(cat_image_url) == True, "Test case failed: Image with a cat should return True."

    # Test case 2: Image without a cat
    no_cat_image_url = 'http://images.cocodataset.org/val2017/000000284665.jpg'
    assert detect_cat_in_image(no_cat_image_url) == False, "Test case failed: Image without a cat should return False."

    print("All tests passed!")

# Run the test function
test_detect_cat_in_image()