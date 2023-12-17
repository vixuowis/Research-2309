# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_cat_dog(image_url):
    # Load the image from URL
    image = Image.open(requests.get(url = image_url, stream = True).raw)

    # Initialize feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')

    # Initialize pre-trained model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Predict the class
    outputs = model(**inputs)
    logits = outputs.logits
    # Considering 0 for 'cat' and 1 for 'dog'
    predicted_class_idx = logits.argmax(-1).item()
    predicted_categories = ['cat', 'dog']

    # Return the prediction
    return predicted_categories[predicted_class_idx]

# test_function_code --------------------

def test_classify_cat_dog():
    print("Testing started.")
    # Test cases using sample images
    test_cases = [
        ('http://example.com/cat_image.jpg', 'cat'),
        ('http://example.com/dog_image.jpg', 'dog')
    ]

    for i, (test_url, expected) in enumerate(test_cases):
        print(f"Testing case [{i+1}/{len(test_cases)}] started.")
        result = classify_cat_dog(test_url)
        assert result == expected, f"Test case [{i+1}/{len(test_cases)}] failed: Expected '{expected}', got '{result}'"
        print(f"Testing case [{i+1}/{len(test_cases)}] successfully passed.")

    print("Testing finished.")