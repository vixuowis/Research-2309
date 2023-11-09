# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify the image from the given URL using the pre-trained model 'google/mobilenet_v1_0.75_192'.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted class of the image.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')
    inputs = preprocessor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    """
    Test the 'classify_image' function with a sample image URL.
    """
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    predicted_class = classify_image(image_url)
    assert isinstance(predicted_class, str), 'The function should return a string.'

# call_test_function_code --------------------

test_classify_image()