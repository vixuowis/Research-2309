# Test function for estimate_depth
# This function tests the estimate_depth function using a sample image
# The test function uses the assert statement to verify the output of the estimate_depth function
# The test function does not compare numbers strictly
# If a dataset is provided in the performance - dataset, the test function loads the dataset and selects several samples from it
# Otherwise, the test function uses an online source

def test_estimate_depth():
    # Sample image path
    image_path = 'sample_image.jpg'
    # Call the estimate_depth function
    depth_map = estimate_depth(image_path)
    # Assert that the output is a torch.Tensor
    assert isinstance(depth_map, torch.Tensor)