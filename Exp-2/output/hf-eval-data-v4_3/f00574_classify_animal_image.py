# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_animal_image(image_path, class_names):
    """
    Classify an image of an animal into one of the provided categories.

    Args:
        image_path (str): The path to the image file to classify.
        class_names (list of str): The list of class names to which the image is to be classified.

    Returns:
        dict: A dictionary with predicted class and its confidence.

    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
    result = classifier(image_path, class_names)
    return result

# test_function_code --------------------

def test_classify_animal_image():
    print("Testing started.")
    # Note: The load_dataset function would typically be imported from a library like datasets or torchvision, depending on the dataset
    sample_data = 'path/to/sample_image.jpg'
    class_names = ['cat', 'dog', 'bird', 'fish']

    # Test case 1: Classify the image correctly
    print("Testing case [1/2] started.")
    result = classify_animal_image(sample_data, class_names)
    assert type(result) is list and len(result) > 0, "Test case [1/2] failed: function did not return expected output type or was empty."

    # Test case 2: Raise FileNotFoundError if the image_path does not exist
    print("Testing case [2/2] started.")
    non_existent_path = 'path/to/non_existent_image.jpg'
    try:
        classify_animal_image(non_existent_path, class_names)
        assert False, "Test case [2/2] failed: FileNotFoundError not raised for non-existent path."
    except FileNotFoundError:
        pass
    print("Testing finished.")


# call_test_function_line --------------------

test_classify_animal_image()