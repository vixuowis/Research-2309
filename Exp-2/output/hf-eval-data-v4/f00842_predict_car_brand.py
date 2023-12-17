# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def predict_car_brand(image_url):
    """
    Predict the car brand from an image URL using a pre-trained image classification model.

    :param image_url: str
        The URL of the image for which to predict the car brand.
    :return: str
        The predicted brand name.
    """
    image = Image.open(requests.get(url, stream=True).raw)
    processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    model = AutoModelForImageClassification.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_predict_car_brand():
    print("Testing predict_car_brand function.")
    # Test case for a known car brand image
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # Replace this URL with an actual car image URL
    predicted_brand = predict_car_brand(test_image_url)
    assert predicted_brand is not None, "Failed to predict car brand."
    print(f"Test passed with prediction: {predicted_brand}")

# Run the test function
test_predict_car_brand()