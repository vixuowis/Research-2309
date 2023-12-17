# requirements_file --------------------

!pip install -U transformers,torch,PIL

# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from PIL import Image

# function_code --------------------

def classify_dog_breed(image_path):
    """
    Classify the dog breed in a given image.

    Parameters:
    image_path (str): The path to the image of the dog to classify.

    Returns:
    str: The classified dog breed.
    """
    # Initialize feature extractor and model
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-tiny-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224')

    # Preprocess the image
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Classify the image using the model
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label_idx = logits.argmax(-1).item()
    dog_breed = model.config.id2label[predicted_label_idx]

    return dog_breed

# test_function_code --------------------

def test_classify_dog_breed():
    print("Testing classify_dog_breed function.")
    test_image_path = 'path/to/dog/image.jpg'  # path to a test dog image

    # Expected output (assuming the test image is a 'Golden Retriever')
    expected_breed = 'Golden Retriever'

    # Test the classify_dog_breed function
    predicted_breed = classify_dog_breed(test_image_path)

    # Assert the prediction is correct
    assert predicted_breed == expected_breed, f"Test failed: expected {expected_breed}, but got {predicted_breed}"