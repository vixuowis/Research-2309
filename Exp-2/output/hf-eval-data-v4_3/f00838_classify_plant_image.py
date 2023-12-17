# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# function_code --------------------

def classify_plant_image(image_path):
    """
    Classify the species of a plant in a given image.

    Args:
        image_path (str): The file path to the image of the plant.

    Returns:
        str: The predicted species of the plant.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image = Image.open(image_path)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_species = model.config.id2label[predicted_class_idx]
    return predicted_species

# test_function_code --------------------

def test_classify_plant_image():
    print("Testing started.")
    test_image_path = 'path_to_test_image.jpg'  # A test image path.

    # Testing case 1: Correct classification
    print("Testing case [1/1] started.")
    predicted_species = classify_plant_image(test_image_path)
    assert predicted_species is not None, f"Test case [1/1] failed: Expected a species classification, got {predicted_species}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_plant_image()