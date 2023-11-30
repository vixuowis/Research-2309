# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth(image_path: str) -> dict:
    """
    Estimate the depth of objects in a given scene using a pre-trained model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The depth estimation result.

    Raises:
        ValueError: If the image_path is not a valid path to an image file.
    """
    
    # check if the input parameter is valid
    if(not (isinstance(image_path, str) and \
            os.path.exists(image_path))):
        raise ValueError("The 'image_path' parameter must be a string pointing to an existing file.") 
        
    # create depth estimation pipeline
    pipe = pipeline('depth-estimation')
    
    # estimate the depth of objects in the scene
    result = pipe(image_path)
    
    return result[0]

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    # Test with a valid image path
    image_path = 'https://placekitten.com/200/300'
    result = estimate_depth(image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'

    # Test with an invalid image path
    try:
        image_path = 'invalid_path'
        result = estimate_depth(image_path)
    except ValueError:
        pass
    else:
        assert False, 'A ValueError should be raised for an invalid image path.'

    print('All Tests Passed')


# call_test_function_code --------------------

test_estimate_depth()