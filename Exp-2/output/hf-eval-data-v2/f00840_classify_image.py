# function_import --------------------

from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify the image using Swin Transformer model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted class of the image.
    """
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Load the pre-trained Swin Transformer model and the feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    # Process the image and pass it to the model
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    # Get the predicted class
    predicted_class_idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    # Use a known image for testing
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    predicted_class = classify_image(image_url)
    # Check if the function returns a string (the class label)
    assert isinstance(predicted_class, str)

# call_test_function_code --------------------

test_classify_image()