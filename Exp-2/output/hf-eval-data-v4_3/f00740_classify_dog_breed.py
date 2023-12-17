# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch

# function_code --------------------

def classify_dog_breed(user_uploaded_image):
    """
    Classifies the dog breed in the given image using a pre-trained ConvNext model.

    Args:
        user_uploaded_image (Image): The image of the dog uploaded by end user.

    Returns:
        str: A string representing the classified dog breed.

    Raises:
        Exception: An exception is raised if the image cannot be processed.
    """
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-tiny-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224')
    inputs = feature_extractor(user_uploaded_image, return_tensors='pt')
    
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    dog_breed = model.config.id2label[predicted_label]
    return dog_breed

# test_function_code --------------------

def test_classify_dog_breed():
    print("Testing started.")
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Load a sample dog image from an online source
    image_url = 'https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg'
    response = requests.get(image_url)
    sample_image = Image.open(BytesIO(response.content))

    # Test case 1: The function should return a string.
    print("Testing case [1/1] started.")
    result = classify_dog_breed(sample_image)
    assert isinstance(result, str), f"Test case [1/1] failed: Expected a string, got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_dog_breed()