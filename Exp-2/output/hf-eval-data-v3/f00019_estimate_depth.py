# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth(image_path: str) -> dict:
    '''
    Estimate depth in an image using a pre-trained model from Hugging Face Transformers.
    
    Args:
    image_path (str): The path to the image file.
    
    Returns:
    dict: A dictionary containing the depth map of the image.
    
    Raises:
    FileNotFoundError: If the image file does not exist.
    '''
    # Import the pipeline function from the transformers library
    from transformers import pipeline
    
    # Use the pipeline function to create a depth estimation model
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-kitti-finetuned-diode-221214-123047')
    
    # Estimate the depth map of the image
    depth_map = depth_estimator(image_path)
    
    return depth_map

# test_function_code --------------------

def test_estimate_depth():
    '''
    Test the estimate_depth function.
    '''
    # Test with a valid image file
    depth_map = estimate_depth('test_image.jpg')
    assert isinstance(depth_map, dict), 'The output should be a dictionary.'
    
    # Test with a non-existent image file
    try:
        depth_map = estimate_depth('non_existent_image.jpg')
    except FileNotFoundError:
        pass
    else:
        assert False, 'A FileNotFoundError should be raised if the image file does not exist.'
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_estimate_depth()