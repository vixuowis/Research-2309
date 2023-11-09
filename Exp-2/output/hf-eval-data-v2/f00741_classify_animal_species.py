# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
from PIL import Image
import torch

# function_code --------------------

def classify_animal_species(animal_image_path):
    """
    Classify different animal species based on their images using a pre-trained RegNet model.

    Args:
        animal_image_path (str): The path to the image of the animal.

    Returns:
        str: The predicted species of the animal.
    """
    # Load the image
    image = Image.open(animal_image_path)

    # Load the pre-trained feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')

    # Preprocess the image
    inputs = feature_extractor(image, return_tensors='pt')

    # Pass the processed image into the model to obtain logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # The category with the highest logits corresponds to the predicted species of the animal
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_animal_species():
    """
    Test the classify_animal_species function.
    """
    # Define a test image path
    test_image_path = 'path_to_test_image'

    # Call the function with the test image
    result = classify_animal_species(test_image_path)

    # Assert that the result is a string (the predicted species)
    assert isinstance(result, str)

# call_test_function_code --------------------

test_classify_animal_species()