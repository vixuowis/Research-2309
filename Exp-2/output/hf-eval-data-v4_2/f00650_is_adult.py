# requirements_file --------------------

!pip install -U requests Pillow transformers

# function_import --------------------

import requests
from PIL import Image
from io import BytesIO
from transformers import ViTFeatureExtractor, ViTForImageClassification

# function_code --------------------

def is_adult(image_url: str) -> bool:
    """
    Determine if the person in the image is an adult based on age classification.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        bool: True if the predicted age class indicates adulthood, False otherwise.

    Raises:
        requests.exceptions.RequestException: If there is a network issue or the image cannot be downloaded.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    inputs = feature_extractor(image, return_tensors='pt')
    output = model(**inputs)

    proba = output.logits.softmax(1)
    predicted_age_class = proba.argmax(1).item()

    # Assuming that the model's class index 1 corresponds to 'adult'
    return predicted_age_class == 1

# test_function_code --------------------

def test_is_adult():
    print("Testing started.")
    # Here we would ideally load images or urls from a dataset
    # since we don't have access to a specific dataset, we use hypothetical URLs
    test_cases = [
        ('https://example.com/adult_image.jpg', True),
        ('https://example.com/non_adult_image.jpg', False),
        ('https://example.com/adult_image2.jpg', True)
    ]

    for i, (url, expected) in enumerate(test_cases, 1):
        print(f"Testing case [{i}/3] started.")
        result = is_adult(url)
        assert result == expected, f"Test case [{i}/3] failed: expected {expected}, got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_is_adult()