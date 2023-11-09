def test_estimate_depth():
    # Load a test image from an online source
    test_image = np.random.rand(224, 224, 3)
    # Call the function with the test image
    test_depth_map = estimate_depth(test_image)
    # Assert that the function returns a depth map with the same dimensions as the input image
    assert test_depth_map.shape == test_image.shape, 'The depth map should have the same dimensions as the input image.'