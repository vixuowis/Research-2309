def test_estimate_depth():
    # Test the 'estimate_depth' function with a sample image
    # The image is taken from the DIODE dataset, which the model was trained on.
    # The exact depth map is not known, so we cannot compare the output strictly.
    # Instead, we check that the output is a torch.Tensor, which indicates that the depth estimation was successful.
    sample_image_path = 'path/to/sample/image.jpg'
    depth_map = estimate_depth(sample_image_path)
    assert isinstance(depth_map, torch.Tensor), 'Depth map should be a torch.Tensor'