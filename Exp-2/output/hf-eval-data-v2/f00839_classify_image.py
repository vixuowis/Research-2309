# function_import --------------------

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image as either a cat or a dog using the pre-trained Vision Transformer (ViT) model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted class of the image, either 'cat' or 'dog'.
    """
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Load the feature extractor and model
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors='pt')
    # Classify the image
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_categories = ['cat', 'dog']
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_categories[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with a sample image URL.
    """
    # Define a sample image URL
    image_url = 'https://example.com/sample.jpg'
    # Call the classify_image function
    result = classify_image(image_url)
    # Assert that the result is either 'cat' or 'dog'
    assert result in ['cat', 'dog'], f'Unexpected result: {result}'

# call_test_function_code --------------------

test_classify_image()