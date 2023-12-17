# requirements_file --------------------

!pip install -U transformers torch PIL requests

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_dog_breed(image_url: str) -> str:
    """
    Classify the breed of a dog in the given image.

    Args:
        image_url (str): The URL of the image to be classified.
    Returns:
        str: The predicted dog breed.
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

def test_classify_dog_breed():
    print("Testing started.")
    image_url = 'https://example.com/test_dog_image.jpg'

    # Test case 1: Check if function returns a type str
    print("Testing case [1/1] started.")
    result = classify_dog_breed(image_url)
    assert isinstance(result, str), f"Test case [1/1] failed: Expected result type 'str', got '{type(result).__name__}'"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_dog_breed()