# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_bean_disease(image_path):
    """
    Detects diseases in bean crops by analyzing an image of the crop leaves.

    Args:
        image_path (str): The file path of the bean leaf image.

    Returns:
        dict: A dictionary containing the predicted disease class and its confidence score.
    """
    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    result = classifier(image_path)
    return result

# test_function_code --------------------

def test_detect_bean_disease():
    print("Testing detect_bean_disease function.")

    # Test case 1: Detecting disease on a known sample image
    print("Testing case [1/1] started.")
    sample_image_path = 'sample_bean_leaf_image.jpg'  # Replace with an actual image path
    prediction = detect_bean_disease(sample_image_path)
    assert type(prediction) is list and len(prediction) > 0, "Test case [1/1] failed: The function did not return a list or returned an empty list."
    assert 'label' in prediction[0], "Test case [1/1] failed: The prediction does not contain a 'label' field."
    assert 'score' in prediction[0], "Test case [1/1] failed: The prediction does not contain a 'score' field."
    print("Testing finished.")

# Run the test function
test_detect_bean_disease()