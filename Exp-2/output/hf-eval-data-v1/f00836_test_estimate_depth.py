def test_estimate_depth():
    """
    Test the estimate_depth function.
    """
    # Define the input image path
    input_image_path = 'test_image.jpg'

    # Call the estimate_depth function
    predicted_depth_map = estimate_depth(input_image_path)

    # Assert the shape of the predicted depth map
    assert predicted_depth_map.shape == (1, 1, 224, 224), 'The shape of the predicted depth map is incorrect.'

    # Assert the type of the predicted depth map
    assert isinstance(predicted_depth_map, torch.Tensor), 'The type of the predicted depth map is incorrect.'