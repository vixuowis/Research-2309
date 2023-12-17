# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image_path, class_names):
    """
    Classify the content of an image into predefined categories.

    Args:
        image_path (str): Path to the image file.
        class_names (list): A list of categories to classify the image into.

    Returns:
        dict: A dictionary containing the predicted category and the confidence score.

    Raises:
        ValueError: If the image path or class names are not provided.
    """
    if not image_path or not class_names:
        raise ValueError('image_path and class_names must be provided.')

    classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
    result = classifier(image_path, class_names=class_names)
    return result

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    # Since we're using an external API, we won't load actual data but mock the classifier response
    mocked_result = [{'label': 'forest', 'score': 0.98}]

    # Test case 1
    print("Testing case [1/3] started.")
    result = classify_image('sample_image_1.jpg', ['landscape', 'cityscape', 'beach', 'forest', 'animals'])
    assert result == mocked_result, f"Test case [1/3] failed: Expected {mocked_result}, got {result}"

    # Test case 2
    print("Testing case [2/3] started.")
    result = classify_image('', [])
    assert result == mocked_result, f"Test case [2/3] failed: Expected {mocked_result}, got {result}"

    # Test case 3
    print("Testing case [3/3] started.")
    result = classify_image('', ['landscape', 'cityscape', 'beach', 'forest', 'animals'])
    assert result == mocked_result, f"Test case [3/3] failed: Expected {mocked_result}, got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()