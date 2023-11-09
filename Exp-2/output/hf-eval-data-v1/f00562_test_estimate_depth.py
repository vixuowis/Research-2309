def test_estimate_depth():
    # Load test image
    test_image = load_image('test_image.jpg')

    # Call the function with the test image
    depth_info = estimate_depth(test_image)

    # Assert the output is not None
    assert depth_info is not None

    # Assert the output is a 2D array (or whatever the expected output format is)
    assert isinstance(depth_info, np.ndarray)
    assert len(depth_info.shape) == 2

    # Assert the output values are within expected range (if applicable)
    assert depth_info.min() >= 0
    assert depth_info.max() <= 1

    print('All tests passed.')

test_estimate_depth()