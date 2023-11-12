# function_import --------------------

from transformers import SwinForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image using a pre-trained Swin Transformer model.

    Args:
        image_url (str): URL of the image to be classified.

    Returns:
        str: Predicted class of the image.

    Raises:
        OSError: If there is a problem with network access or disk usage.
    """
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Load the pre-trained Swin Transformer model
    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')

    # Process the image and return tensors
    inputs = model.feature_extractor(images=image, return_tensors='pt')

    # Pass the image features to the model for classification
    outputs = model(**inputs)

    # Get the predicted class index
    predicted_class_idx = outputs.logits.argmax(-1).item()

    # Return the predicted class label
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with different test cases.
    """
    # Test case 1: A URL of a cat image
    image_url = 'https://placekitten.com/200/300'
    predicted_class = classify_image(image_url)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'

    # Test case 2: A URL of a dog image
    image_url = 'https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg'
    predicted_class = classify_image(image_url)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'

    # Test case 3: A URL of a bird image
    image_url = 'https://www.audubon.org/sites/default/files/styles/hero_cover_bird_page/public/web_bird-american-robin-1280x720.jpg?itok=7S6zTlnK'
    predicted_class = classify_image(image_url)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()