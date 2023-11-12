# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def predict_dog_breed(image_url):
    """
    Predict the breed of a dog from an image.

    Args:
        image_url (str): The URL of the image to predict.

    Returns:
        str: The predicted breed of the dog.

    Raises:
        PIL.UnidentifiedImageError: If the image cannot be identified.
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
    """Test the predict_dog_breed function."""
    assert predict_dog_breed('https://placekitten.com/200/300') is not None
    assert isinstance(predict_dog_breed('https://placekitten.com/200/300'), str)
    try:
        predict_dog_breed('https://example.com/non_existent_image.jpg')
    except Exception as e:
        assert isinstance(e, PIL.UnidentifiedImageError)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_dog_breed()