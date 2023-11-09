# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_houseplant(url):
    """
    Classify the type of houseplant in an image.

    Args:
        url (str): The URL of the image to classify.

    Returns:
        str: The predicted type of the houseplant.
    """
    image = Image.open(requests.get(url, stream=True).raw)
    preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')
    inputs = preprocessor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_houseplant():
    """
    Test the classify_houseplant function.
    """
    url = 'https://example.com/houseplant_image.jpg'
    result = classify_houseplant(url)
    assert isinstance(result, str)

# call_test_function_code --------------------

test_classify_houseplant()