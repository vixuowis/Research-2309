# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image_path: str, class_names: list) -> dict:
    '''
    Classify an image into predefined categories using a pre-trained model.

    Args:
        image_path (str): The path to the image to be classified.
        class_names (list): The list of categories to classify the image into.

    Returns:
        dict: A dictionary containing the probabilities of the image belonging to each category.

    Raises:
        ValueError: If the image_path is not a string or if class_names is not a list.
        FileNotFoundError: If the image at image_path does not exist.
    '''
    if not isinstance(image_path, str):
        raise ValueError('image_path must be a string')
    if not isinstance(class_names, list):
        raise ValueError('class_names must be a list')
    clip = pipeline('image-classification', model='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
    result = clip(image_path, class_names=class_names)
    return result

# test_function_code --------------------

def test_classify_image():
    '''
    Test the classify_image function.
    '''
    # Test with valid inputs
    result = classify_image('https://placekitten.com/200/300', ['landscape', 'cityscape', 'beach', 'forest', 'animals'])
    assert isinstance(result, dict), 'Result should be a dictionary.'
    assert all(isinstance(key, str) for key in result.keys()), 'All keys in result should be strings.'
    assert all(isinstance(value, float) for value in result.values()), 'All values in result should be floats.'
    # Test with invalid inputs
    try:
        classify_image(123, ['landscape', 'cityscape', 'beach', 'forest', 'animals'])
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError when image_path is not a string.'
    try:
        classify_image('https://placekitten.com/200/300', 'landscape, cityscape, beach, forest, animals')
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError when class_names is not a list.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()