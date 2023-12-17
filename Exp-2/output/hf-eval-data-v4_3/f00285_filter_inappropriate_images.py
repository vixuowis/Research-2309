# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def filter_inappropriate_images(image_path):
    """
    Processes an image and determines whether it contains adult content or offensive
    material using zero-shot classification.

    Args:
        image_path (str): The file path or URL of the image to be classified.

    Returns:
        dict: Classification results with category probabilities.

    Raises:
        ValueError: If the image_path is None or an empty string.
    """
    if not image_path:
        raise ValueError('The image_path argument must be a valid non-empty string.')
    image_classifier = pipeline('zero-shot-classification', model='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    class_names = ['safe for work', 'adult content', 'offensive']
    result = image_classifier(image=image_path, class_names=class_names)
    return result

# test_function_code --------------------

def test_filter_inappropriate_images():
    print("Testing started.")
    image_sample_safe = 'path/to/safe_image'
    image_sample_adult = 'path/to/adult_image'
    image_sample_offensive = 'path/to/offensive_image'

    print("Testing case [1/3] started.")
    result_safe = filter_inappropriate_images(image_sample_safe)
    assert 'safe for work' in result_safe['labels'], "Test case [1/3] failed: Image misclassified as unsafe or offensive."

    print("Testing case [2/3] started.")
    result_adult = filter_inappropriate_images(image_sample_adult)
    assert 'adult content' in result_adult['labels'], "Test case [2/3] failed: Inappropriate adult content not identified."

    print("Testing case [3/3] started.")
    result_offensive = filter_inappropriate_images(image_sample_offensive)
    assert 'offensive' in result_offensive['labels'], "Test case [3/3] failed: Offensive content not recognized."
    print("Testing finished.")

# call_test_function_line --------------------

test_filter_inappropriate_images()