# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def predict_dog_breed(image_url):
    """
    Recognize the breed of dog in the given image using Vision Transformer (ViT) model.

    Args:
        image_url (str): URL of the image to be classified.

    Returns:
        str: Predicted breed of the dog in the image.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_predict_dog_breed():
    """
    Test the function predict_dog_breed.
    """
    image_url = 'https://example.com/dog_image.jpg'
    predicted_breed = predict_dog_breed(image_url)
    assert isinstance(predicted_breed, str), 'The output should be a string.'
    assert predicted_breed != '', 'The output should not be an empty string.'

# call_test_function_code --------------------

test_predict_dog_breed()