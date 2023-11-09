def test_estimate_depth():
    '''
    This function tests the 'estimate_depth' function.
    It uses a test image and checks if the output is not None.
    '''
    # Path to the test image
    test_image_path = 'test_image.jpg'

    # Call the 'estimate_depth' function with the test image
    depth_estimation = estimate_depth(test_image_path)

    # Check if the output is not None
    assert depth_estimation is not None, 'The depth estimation should not be None.'

    # Check if the output is a tensor
    assert isinstance(depth_estimation, torch.Tensor), 'The output should be a tensor.'

    # Check if the output has the correct shape
    assert len(depth_estimation.shape) == 4, 'The output tensor should have 4 dimensions.'

    # Call the test function
    test_estimate_depth()