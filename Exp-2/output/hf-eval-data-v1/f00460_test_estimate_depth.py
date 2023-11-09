def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    It uses a sample image from the internet.
    """
    # Define the URL of the test image
    test_image_url = 'https://example.com/test_image.jpg'

    # Call the function with the test image
    depth_map = estimate_depth(test_image_url)

    # Check the type of the output
    assert isinstance(depth_map, torch.Tensor), 'Output should be a torch.Tensor'

    # Check the shape of the output
    assert len(depth_map.shape) == 4, 'Output should have 4 dimensions'

    # Check the values of the output
    assert torch.all(depth_map >= 0), 'All values should be non-negative'

    print('All tests passed.')

# Run the test function
test_estimate_depth()