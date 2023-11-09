# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch

# function_code --------------------

def classify_image(image_url):
    '''
    Classify an image using the Vision Transformer (ViT) model.

    Args:
        image_url (str): URL of the image to be classified.

    Returns:
        str: Predicted class of the image.
    '''
    # Open the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Instantiate the image processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    # Load the pre-trained Vision Transformer model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # Pre-process the image
    inputs = processor(images=image, return_tensors='pt')
    # Perform image classification
    outputs = model(**inputs)
    logits = outputs.logits
    # Get the predicted class index
    predicted_class_idx = logits.argmax(-1).item()
    # Return the predicted class
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    '''
    Test the classify_image function.
    '''
    # Define a test image URL
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # Call the classify_image function
    predicted_class = classify_image(test_image_url)
    # Assert that the function returns a string (the predicted class)
    assert isinstance(predicted_class, str)

# call_test_function_code --------------------

test_classify_image()