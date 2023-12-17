# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_food_image(image_path, possible_class_names=None):
    """
    Classifies the food image into predefined classes using a zero-shot image classification approach.

    :param image_path: Path to the image file to be classified.
    :param possible_class_names: A list of string representing possible food categories for classification.
    :return: Classification result as a dictionary containing labels and scores.
    """
    # Create an image classifier model with the specified pre-trained model
    image_classifier = pipeline('image-classification', model='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    # If no possible classes are provided, use a default list
    if not possible_class_names:
        possible_class_names = ['pizza', 'sushi', 'sandwich', 'salad', 'cake']

    # Classify the image
    result = image_classifier(image_path, possible_class_names=possible_class_names)
    return result

# test_function_code --------------------

def test_classify_food_image():
    print("Testing classify_food_image function started.")

    # This is a placeholder path. Replace 'sample_image.jpg' with a real image path when testing.
    sample_image_path = 'sample_image.jpg'

    # Test case 1: Classify with default categories
    print("Testing case [1/2] started.")
    result_default = classify_food_image(sample_image_path)
    assert len(result_default) > 0, f"Test case [1/2] failed: Expected non-empty result, got {result_default}"

    # Test case 2: Classify with custom categories
    print("Testing case [2/2] started.")
    custom_categories = ['burger', 'taco', 'pasta']
    result_custom = classify_food_image(sample_image_path, possible_class_names=custom_categories)
    assert len(result_custom) > 0, f"Test case [2/2] failed: Expected non-empty result, got {result_custom}"
    print("Testing classify_food_image function finished.")