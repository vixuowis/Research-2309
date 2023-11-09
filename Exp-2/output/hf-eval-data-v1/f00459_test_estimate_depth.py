def test_estimate_depth():
    """
    This function tests the 'estimate_depth' function.
    It uses a sample image from an online source.
    """
    # Define the path to the sample image
    sample_image_path = 'https://example.com/sample_image.jpg'
    
    # Call the 'estimate_depth' function
    depth_map = estimate_depth(sample_image_path)
    
    # Check the type of the output
    assert isinstance(depth_map, np.ndarray), 'The output should be a numpy array.'
    
    # Check the shape of the output
    assert len(depth_map.shape) == 2, 'The output should be a 2D array.'
    
    # Check the values of the output
    assert np.all(depth_map >= 0), 'All values in the depth map should be non-negative.'

test_estimate_depth()