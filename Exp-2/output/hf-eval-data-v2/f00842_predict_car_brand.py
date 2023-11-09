# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def predict_car_brand(image_url):
    """
    This function predicts the car brand from an image using a pre-trained model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted car brand.
    """
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Load the pre-trained model and the processor
    processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    model = AutoModelForImageClassification.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    # Process the image and predict the class
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    # Return the predicted class
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_predict_car_brand():
    """
    This function tests the predict_car_brand function.
    It uses a known image of a car and checks if the predicted brand is in the list of possible brands.
    """
    # Known image of a car
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # List of possible car brands
    possible_brands = ['Audi', 'BMW', 'Mercedes-Benz', 'Volkswagen', 'Porsche']
    # Predict the car brand
    predicted_brand = predict_car_brand(image_url)
    # Check if the predicted brand is in the list of possible brands
    assert predicted_brand in possible_brands, f'Expected one of {possible_brands}, but got {predicted_brand}'

# call_test_function_code --------------------

test_predict_car_brand()