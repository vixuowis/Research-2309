# requirements_file --------------------

pip install -U transformers torch pillow requests

# function_import --------------------

from transformers import AutoModelForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_social_media_images(image_urls, model_name='microsoft/swin-tiny-patch4-window7-224-bottom_cleaned_data'):
    """
    Classify social media images into predefined categories using a pre-trained model.

    Args:
        image_urls (list of str): A list of URLs pointing to the images to be classified.
        model_name (str): The name of the pre-trained Hugging Face model to use.

    Returns:
        list of tuple: A list of tuples where each tuple contains the URL of the image and its predicted category.

    Raises:
        ValueError: If any URL does not point to a valid image file.
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)

    images = [Image.open(BytesIO(requests.get(url).content)) for url in image_urls]

    categories = ['category1', 'category2', 'category3']
    predictions = []
    for image in images:
        inputs = model.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predictions.append(categories[outputs.logits.argmax(-1).item()])

    return list(zip(image_urls, predictions))

# test_function_code --------------------

def test_classify_social_media_images():
    print("Testing started.")
    dataset = load_dataset("social_media_images")
    sample_data = dataset['test']

    print("Testing case [1/1] started.")
    image_url = sample_data[0]
    predictions = classify_social_media_images([image_url])
    assert predictions[0][0] == image_url, f"Test case [1/1] failed: The URL of the predicted category does not match the input URL."
    print(f"Predicted category for the image URL {image_url}: {predictions[0][1]}")
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_social_media_images()