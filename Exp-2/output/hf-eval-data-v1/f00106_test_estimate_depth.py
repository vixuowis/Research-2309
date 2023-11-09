# Test function for the 'estimate_depth' function
# The function takes in no arguments
# It uses a test image to check the functionality of the 'estimate_depth' function
# The test image is assumed to be located at 'test_street_image.jpg'
# The function asserts that the output of the 'estimate_depth' function is a torch.Tensor

def test_estimate_depth():
    test_image_path = 'test_street_image.jpg'
    # Replace 'test_street_image.jpg' with the actual path to your test image
    depth_map = estimate_depth(test_image_path)
    assert isinstance(depth_map, torch.Tensor), 'Output should be a torch.Tensor'