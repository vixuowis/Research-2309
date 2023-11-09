def test_estimate_depth():
    """
    This function tests the estimate_depth function.
    It uses a sample image and checks if the output image is correctly saved.
    """
    import os
    estimate_depth('sample_input.png', 'test_output.png')
    assert os.path.exists('test_output.png'), 'Output image not found.'