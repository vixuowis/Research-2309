# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_image(image_url: str) -> str:
    """
    Classify an image using the pretrained RegNetForImageClassification model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted label of the image.

    Raises:
        OSError: If the model identifier is not found in the Hugging Face model hub.
    """
    model = RegNetForImageClassification.from_pretrained('facebook/regnet-y-008')
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/regnet-y-008')

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    inputs = feature_extractor(images=image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with different test cases.
    """
    test_image_url_1 = 'https://placekitten.com/200/300'
    test_image_url_2 = 'https://placekitten.com/400/600'
    test_image_url_3 = 'https://placekitten.com/800/1200'

    assert isinstance(classify_image(test_image_url_1), str)
    assert isinstance(classify_image(test_image_url_2), str)
    assert isinstance(classify_image(test_image_url_3), str)

    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_image()