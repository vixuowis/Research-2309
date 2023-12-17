# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch", "PIL"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline
import PIL.Image

# function_code --------------------

def classify_image(image_path):
    """
    Classifies an image using a pre-trained ViT model.

    Args:
        image_path (str): The path to the image file to classify.

    Returns:
        list: A list of dictionaries containing the predicted categories and associated probabilities.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        Exception: If the classification pipeline encounters an error.
    """
    # Check if image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The provided image path does not exist: {image_path}")

    # Load and classify the image
    try:
        image_classifier = pipeline('image-classification', model='timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k', framework='pt')
        image = PIL.Image.open(image_path)
        result = image_classifier(image)
        return result
    except Exception as e:
        raise e

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    image_path = 'path/to/your/test/image.jpg'  # Replace with actual image path

    # Test case 1: Valid image file
    print("Testing case [1/2] started.")
    result_1 = classify_image(image_path)
    assert isinstance(result_1, list) and len(result_1) > 0, f"Test case [1/2] failed: {result_1}"

    # Test case 2: Non-existing image file
    print("Testing case [2/2] started.")
    try:
        result_2 = classify_image('non_existing_image.jpg')
    except FileNotFoundError:
        assert True
    else:
        raise AssertionError(f"Test case [2/2] failed: {result_2}")
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()