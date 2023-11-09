def test_estimate_depth():
    # Test the estimate_depth function
    # Load a test image
    test_image_path = 'test_image.jpg'
    
    # Call the function with the test image
    depth_estimates = estimate_depth(test_image_path)
    
    # Check that the output is a tensor
    assert isinstance(depth_estimates, torch.Tensor), 'Output should be a tensor'
    
    # Check that the output has the correct shape
    assert depth_estimates.shape == (1, 1, 480, 640), 'Output shape should be (1, 1, 480, 640)'
    
    # Check that the output values are reasonable
    assert depth_estimates.min() >= 0, 'Depth estimates should be non-negative'
    assert depth_estimates.max() <= 100, 'Depth estimates should be less than or equal to 100'

test_estimate_depth()