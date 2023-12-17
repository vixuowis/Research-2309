# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "datasets", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
from PIL import Image
import torch

# function_code --------------------

def classify_animal_species(image_path):
    """
    Classify the species of an animal based on its image.

    Args:
        image_path (str): The file path to the image of the animal.

    Returns:
        str: The predicted species label of the animal.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If there is an error during model inference.
    """
    # Load and preprocess the image
    image = Image.open(image_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    inputs = feature_extractor(image, return_tensors='pt')

    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# test_function_code --------------------

from datasets import load_dataset

def test_classify_animal_species():
    print("Testing started.")
    dataset = load_dataset("huggingface/cats-image", split='test')
    sample_data = dataset[0]  # Get a sample image file

    # Test case 1
    print("Testing case [1/1] started.")
    try:
        result = classify_animal_species(sample_data['image'])
        assert isinstance(result, str), f"Test case [1/1] failed: Expected str, got {type(result)}"
        print("Test case [1/1] success.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_animal_species()