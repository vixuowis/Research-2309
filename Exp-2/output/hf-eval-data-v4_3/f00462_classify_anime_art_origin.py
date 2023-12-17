# requirements_file --------------------

import subprocess

requirements = ["transformers", "Pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline
from PIL import Image

# function_code --------------------

def classify_anime_art_origin(image_path):
    """
    Classifies the origin of anime art by determining whether it was created by a human or AI.

    Args:
        image_path (str): The path to the image file to be classified.

    Returns:
        dict: A dictionary containing the classification result and label.

    Raises:
        FileNotFoundError: If the image file at the specified path does not exist.
    """
    # Load the image
    try:
        image = Image.open(image_path)
    except FileNotFoundError as e:
        raise FileNotFoundError("Image file not found.") from e

    # Initialize the image classification model
    anime_detector = pipeline('image-classification', model='saltacc/anime-ai-detect')

    # Perform classification
    classification_result = anime_detector(image)
    return classification_result

# test_function_code --------------------

def test_classify_anime_art_origin():
    print("Testing started.")
    # Assuming 'test_image.jpg' is a valid image file in the current directory
    test_image_path = 'test_image.jpg'

    # Testing correct classification
    print("Testing case [1/1] started.")
    result = classify_anime_art_origin(test_image_path)
    assert isinstance(result, dict), f"Test case [1/1] failed: Expected result to be a dictionary, got {type(result).__name__} instead."
    assert 'label' in result, f"Test case [1/1] failed: Expected key 'label' in result, not found."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_anime_art_origin()