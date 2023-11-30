# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image(img_url: str):
    """
    Classify an image using a pretrained CLIP model.

    Args:
        img_url (str): The URL of the image to classify.

    Returns:
        dict: A dictionary where keys are labels and values are probabilities.
    """

    # Download Image --------------------
    image = Image.open(requests.get(img_url, stream=True).raw)

    # Process Image --------------------
    image = preprocess(image)

    # Predict Image --------------------
    model = load_model()
    text = predict(image, model)

    return text


# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    img_url = 'https://placekitten.com/200/300'
    result = classify_image(img_url)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(['residential area', 'playground', 'stadium', 'forest', 'airport'])
    assert all(0 <= v <= 1 for v in result.values())
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_image()