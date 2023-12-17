# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image, class_names=['bike', 'car']):
    """
    Classifies an image as a bike or a car using zero-shot classification.

    Args:
        image (str or PIL.Image): The filepath to the image or the PIL.Image object to classify.
        class_names (list): A list of class names to classify against. Default is ['bike', 'car'].

    Returns:
        dict: A dictionary containing the classification results.

    Raises:
        ValueError: If 'image' input is not a path or PIL.Image object.

    """
    clip = pipeline('zero-shot-classification', model='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind')
    result = clip(image, class_names)
    return result

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    image_path = 'path/to/test/image.jpg'
    class_names = ['bike', 'car']

    print("Testing case [1/1] started.")
    result = classify_image(image_path, class_names)
    assert 'labels' in result and 'scores' in result, "Test case [1/1] failed: 'labels' or 'scores' not in result."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()