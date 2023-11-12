# function_import --------------------

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_food_images(image_url: str, possible_class_names: list = ['pizza', 'sushi', 'sandwich', 'salad', 'cake']) -> dict:
    """
    Classify food images using a pre-trained model from Hugging Face Transformers.

    Args:
        image_url (str): The URL of the image to be classified.
        possible_class_names (list, optional): A list of possible food classes. Defaults to ['pizza', 'sushi', 'sandwich', 'salad', 'cake'].

    Returns:
        dict: The classification results.
    """
    # Load the pre-trained model
    image_classifier = pipeline('image-classification', model='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Classify the image
    result = image_classifier(image, possible_class_names=possible_class_names)

    return result

# test_function_code --------------------

def test_classify_food_images():
    """
    Test the classify_food_images function.
    """
    # Test case: Classify a pizza image
    pizza_url = 'https://example.com/pizza.jpg'
    result = classify_food_images(pizza_url)
    assert 'pizza' in [item['label'] for item in result], 'Test case 1 failed'

    # Test case: Classify a sushi image
    sushi_url = 'https://example.com/sushi.jpg'
    result = classify_food_images(sushi_url)
    assert 'sushi' in [item['label'] for item in result], 'Test case 2 failed'

    # Test case: Classify a sandwich image
    sandwich_url = 'https://example.com/sandwich.jpg'
    result = classify_food_images(sandwich_url)
    assert 'sandwich' in [item['label'] for item in result], 'Test case 3 failed'

    return 'All tests passed'

# call_test_function_code --------------------

test_classify_food_images()