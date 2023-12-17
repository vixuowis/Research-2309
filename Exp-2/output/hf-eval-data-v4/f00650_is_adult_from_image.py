# requirements_file --------------------

!pip install -U requests Pillow transformers

# function_import --------------------

import requests
from PIL import Image
from io import BytesIO
from transformers import ViTFeatureExtractor, ViTForImageClassification

# function_code --------------------

def is_adult_from_image(image_url):
    """
    Check if the person in the provided image URL is classified as an adult based on age.
    :param image_url: URL of the image to be classified
    :return: boolean indicating if the person is an adult
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
    inputs = transforms(image, return_tensors='pt')
    output = model(**inputs)
    probabilities = output.logits.softmax(1)
    predicted_age_class_idx = probabilities.argmax(1)
    ADULT_AGE_CLASS_IDX = 4  # Assuming classifier's index 4 indicates adulthood
    return predicted_age_class_idx.item() >= ADULT_AGE_CLASS_IDX

# test_function_code --------------------

def test_is_adult_from_image():
    print("Testing started.")
    # Test image URLs
    adult_image_url = 'https://some-adult-image-url.jpg'
    non_adult_image_url = 'https://some-non-adult-image-url.jpg'

    print("Testing adult case started.")
    assert is_adult_from_image(adult_image_url), f"Test adult case failed: is_adult_from_image({adult_image_url}) is not True."

    print("Testing non-adult case started.")
    assert not is_adult_from_image(non_adult_image_url), f"Test non-adult case failed: is_adult_from_image({non_adult_image_url}) is not False."
    print("Testing finished.")

# Run the test function
test_is_adult_from_image()