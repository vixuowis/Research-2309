# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
from PIL import Image
import torch

# function_code --------------------

def classify_animal_species(animal_image_path):
    """
    Classify the species of an animal based on its image using a pre-trained RegNet model.

    Args:
        animal_image_path (str): The path to the image of the animal.

    Returns:
        str: The predicted species of the animal.

    Raises:
        FileNotFoundError: If the image file does not exist.
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
    """
    Test the classify_animal_species function with a few test cases.
    """
    test_image_path = 'https://placekitten.com/200/300'
    result = classify_animal_species(test_image_path)
    assert isinstance(result, str), 'The result should be a string.'
    assert result in model.config.id2label.values(), 'The result should be a valid label.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_animal_species()