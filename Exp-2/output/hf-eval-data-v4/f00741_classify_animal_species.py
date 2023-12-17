# requirements_file --------------------

!pip install -U transformers torch pillow

# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image

# function_code --------------------

def classify_animal_species(animal_image_path):
    """
    Classify an image of an animal into its species using the RegNet model.

    Parameters:
    animal_image_path (str): The file path to the image of the animal.

    Returns:
    str: The identified species label of the animal.
    """
    image = Image.open(animal_image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_animal_species():
    print("Testing started.")
    sample_image_path = 'path/to/sample_image.jpg'  # A path to a sample image

    # Test case 1: Ensure function returns a string
    print("Testing case [1/1] started.")
    result = classify_animal_species(sample_image_path)
    assert isinstance(result, str), f"Test case [1/1] failed: Expected result to be a string, got {type(result)}"
    print("Testing finished.")