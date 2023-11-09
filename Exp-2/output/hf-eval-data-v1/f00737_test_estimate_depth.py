def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    It uses a sample image from an online source.
    """
    # Define the path to the sample image
    sample_image_path = 'https://example.com/sample_image.jpg'
    
    # Call the function with the sample image
    depth_pred = estimate_depth(sample_image_path)
    
    # Assert that the function returns a tensor
    assert isinstance(depth_pred, torch.Tensor), 'The function should return a tensor.'
    
    # Assert that the tensor has the correct shape
    assert depth_pred.shape == (1, 1, 480, 640), 'The tensor should have shape (1, 1, 480, 640).'

test_estimate_depth()