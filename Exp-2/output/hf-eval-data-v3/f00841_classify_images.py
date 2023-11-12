# function_import --------------------

from transformers import AutoModelForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_images(image_urls, categories):
    """
    Classify images into various categories using a pre-trained model from Hugging Face Transformers.

    Args:
        image_urls (list of str): List of URLs of the images to be classified.
        categories (list of str): List of categories that the images can be classified into.

    Returns:
        dict: A dictionary where keys are image URLs and values are their corresponding categories.
    """
    model = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224-bottom_cleaned_data')
    results = {}
    for image_url in image_urls:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        results[image_url] = categories[predicted.item()]
    return results

# test_function_code --------------------

def test_classify_images():
    """
    Test the classify_images function.
    """
    image_urls = ['https://placekitten.com/200/300', 'https://placekitten.com/200/301', 'https://placekitten.com/200/302']
    categories = ['cat', 'dog', 'bird']
    results = classify_images(image_urls, categories)
    assert isinstance(results, dict)
    assert len(results) == len(image_urls)
    for image_url, category in results.items():
        assert image_url in image_urls
        assert category in categories
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_images()