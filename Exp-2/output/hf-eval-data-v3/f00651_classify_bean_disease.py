# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_bean_disease(image_path):
    """
    Classify the disease of a bean crop based on an image of the crop leaf.

    Args:
        image_path (str): The path to the image of the bean crop leaf.

    Returns:
        dict: The predicted disease of the bean crop.

    Raises:
        OSError: If the image file cannot be found or read.
    """
    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    result = classifier(image_path)
    return result

# test_function_code --------------------

def test_classify_bean_disease():
    """
    Test the classify_bean_disease function.
    """
    test_image_path = 'path/to/test_image.jpg'
    result = classify_bean_disease(test_image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should contain a label.'
    assert 'score' in result, 'The result should contain a score.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_bean_disease()