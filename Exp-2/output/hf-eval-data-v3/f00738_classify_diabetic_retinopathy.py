# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_diabetic_retinopathy(image_path):
    """
    Classify whether the given image indicates diabetic retinopathy using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The classification result.
    """
    image_classifier = pipeline('image-classification', 'martinezomg/vit-base-patch16-224-diabetic-retinopathy')
    result = image_classifier(image_path)
    return result

# test_function_code --------------------

def test_classify_diabetic_retinopathy():
    """
    Test the classify_diabetic_retinopathy function.
    """
    test_image_path = 'https://placekitten.com/200/300'
    result = classify_diabetic_retinopathy(test_image_path)
    assert isinstance(result, list), 'The result should be a list.'
    assert isinstance(result[0], dict), 'Each item in the result should be a dictionary.'
    assert 'label' in result[0], 'Each item in the result should have a label.'
    assert 'score' in result[0], 'Each item in the result should have a score.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_diabetic_retinopathy()