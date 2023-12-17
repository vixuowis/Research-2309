# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_product_image(image_path, class_names):
    """Classify the product image into specified categories.

    Args:
        image_path (str): The path to the image file that needs to be classified.
        class_names (list of str): The list of class names to be used for classification.

    Returns:
        dict: The predicted class and its probability.

    Raises:
        ValueError: If the image_path does not exist.
    """
    # Check if the image path exists
    if not os.path.exists(image_path):
        raise ValueError('The image path does not exist.')

    # Initialize the classifier using the specified model
    device_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')

    # Perform classification
    prediction = device_classifier(image_path, class_names)
    return prediction

# test_function_code --------------------

def test_classify_product_image():
    print("Testing started.")
    image_path = 'tests/assets/product_sample.jpg'  # This path should contain a test image
    class_names = ['smartphone', 'laptop', 'tablet']

    # Testing case 1: Valid image path
    print("Testing case [1/1] started.")
    prediction = classify_product_image(image_path, class_names)
    assert isinstance(prediction, dict) and 'label' in prediction, f"Test case [1/1] failed: Expected a dictionary with label, got {prediction}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_product_image()