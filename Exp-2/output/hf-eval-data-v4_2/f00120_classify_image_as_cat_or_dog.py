# requirements_file --------------------

!pip install -U pillow requests transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image_as_cat_or_dog(image_url):
    """
    Classify an image as a cat or a dog using zero-shot classification.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The classification result, either 'cat' or 'dog'.

    Raises:
        ValueError: If the URL is inaccessible or not an image.
    """
    # Load the pre-trained CLIP model
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # Try to open the image from URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Unable to access image from the provided URL.') from e

    # Process the image and the text labels
    inputs = processor(
        text=['a photo of a cat', 'a photo of a dog'],
        images=image,
        return_tensors='pt',
        padding=True)

    # Model prediction
    outputs = model(**inputs)

    # Calculate probabilities and return the classification result
    probs = outputs.logits_per_image.softmax(dim=1)
    classification = 'cat' if probs[0, 0] > probs[0, 1] else 'dog'
    return classification


# test_function_code --------------------

def test_classify_image_as_cat_or_dog():
    print("Testing started.")
    
    # Test image URLs
    cat_image_url = 'https://example.com/cat.jpg'  # Replace with an actual cat image URL
    dog_image_url = 'https://example.com/dog.jpg'  # Replace with an actual dog image URL

    # Test case 1: Cat image
    print("Testing case [1/2] started.")
    result_cat = classify_image_as_cat_or_dog(cat_image_url)
    assert result_cat == 'cat', f"Test case [1/2] failed: Expected 'cat', got {result_cat}"

    # Test case 2: Dog image
    print("Testing case [2/2] started.")
    result_dog = classify_image_as_cat_or_dog(dog_image_url)
    assert result_dog == 'dog', f"Test case [2/2] failed: Expected 'dog', got {result_dog}"

    print("Testing finished.")


# call_test_function_line --------------------

test_classify_image_as_cat_or_dog()