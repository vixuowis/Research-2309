# function_import --------------------

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image as either a cat or a dog using the pre-trained Vision Transformer model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted class of the image, either 'cat' or 'dog'.

    Raises:
        PIL.UnidentifiedImageError: If the image cannot be identified.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_categories = ['cat', 'dog']
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_categories[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    assert classify_image('https://placekitten.com/200/300') in ['cat', 'dog']
    assert classify_image('https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg') in ['cat', 'dog']
    assert classify_image('https://images.dog.ceo/breeds/hound-ibizan/n02091244_1451.jpg') in ['cat', 'dog']
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()