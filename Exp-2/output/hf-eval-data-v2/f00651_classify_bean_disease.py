# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_bean_disease(image_path):
    """
    This function classifies the disease of a bean crop based on an image of the crop leaf.

    Args:
        image_path (str): The path to the image of the bean crop leaf.

    Returns:
        dict: The predicted disease of the bean crop.
    """
    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    result = classifier(image_path)
    return result

# test_function_code --------------------

def test_classify_bean_disease():
    """
    This function tests the classify_bean_disease function by using a sample image of a bean crop leaf.
    """
    test_image_path = 'path/to/test_image.jpg'
    result = classify_bean_disease(test_image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should have a label.'
    assert 'score' in result, 'The result should have a score.'

# call_test_function_code --------------------

test_classify_bean_disease()