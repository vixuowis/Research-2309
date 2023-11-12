# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_animal(image_path: str) -> str:
    """
    Classify an animal in an image as a cat or a dog using a pre-trained model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The predicted class of the animal in the image ('cat' or 'dog').

    Raises:
        OSError: If the specified model is not found.
    """
    try:
        image_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
        result = image_classifier(image_path, ['cat', 'dog'])
        return result[0]['label']
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_classify_animal():
    """
    Test the classify_animal function with different test cases.
    """
    # Test case 1: Image of a cat
    assert classify_animal('https://placekitten.com/200/300') == 'cat'
    # Test case 2: Image of a dog
    assert classify_animal('https://placedog.net/500') == 'dog'
    # Test case 3: Non-existent model
    try:
        classify_animal('https://placekitten.com/200/300', model='non-existent-model')
    except OSError:
        pass
    else:
        raise AssertionError('Expected an OSError for a non-existent model.')
    print('All Tests Passed')

# call_test_function_code --------------------

test_classify_animal()