# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def predict_dog_breed(image_url):
    """
    Predict the breed of a dog from an image URL.

    Args:
        image_url (str): The URL of the image to predict.

    Returns:
        str: The predicted breed of the dog.

    Raises:
        OSError: If there is a problem with the network connection or the image file.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-tiny-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224')
    inputs = feature_extractor(img, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    dog_breed = model.config.id2label[predicted_label]
    return dog_breed

# test_function_code --------------------

def test_predict_dog_breed():
    """
    Test the predict_dog_breed function.
    """
    assert predict_dog_breed('https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg') == 'Afghan_hound'
    assert predict_dog_breed('https://images.dog.ceo/breeds/retriever-golden/n02099601_1.jpg') == 'Golden_retriever'
    assert predict_dog_breed('https://images.dog.ceo/breeds/spaniel-blenheim/n02086646_103.jpg') == 'Blenheim_spaniel'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_dog_breed()