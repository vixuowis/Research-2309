def test_estimate_depth():
    # Path to a test image.
    test_image_path = 'path/to/test/image.jpg'

    # Call the function with the test image.
    estimated_depth = estimate_depth(test_image_path)

    # Check if the function returns a tensor.
    assert isinstance(estimated_depth, torch.Tensor), 'The function should return a torch.Tensor.'

    # Check if the tensor has the correct shape (1, Height, Width).
    # Note: The exact shape will depend on the size of the input image.
    assert len(estimated_depth.shape) == 3 and estimated_depth.shape[0] == 1, 'The returned tensor should have shape (1, Height, Width).'

    # Check if the values in the tensor are within a reasonable range (0-255).
    # Note: The exact range may vary depending on the model and the input image.
    assert torch.all(0 <= estimated_depth) and torch.all(estimated_depth <= 255), 'The values in the returned tensor should be in the range 0-255.'

test_estimate_depth()