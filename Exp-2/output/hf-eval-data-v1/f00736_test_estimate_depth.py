import numpy as np

# Test function for estimate_depth

def test_estimate_depth():
    """
    Test function for estimate_depth. Uses a sample image to test the function.
    """
    # Path to a sample image
    image_path = 'sample_image.jpg'
    
    # Call the function with the sample image
    depth_map = estimate_depth(image_path)
    
    # Check that the output is a numpy array
    assert isinstance(depth_map, np.ndarray), 'Output should be a numpy array'
    
    # Check that the output array is not empty
    assert depth_map.size > 0, 'Output array should not be empty'
    
    # Call the function again with a different image
    depth_map2 = estimate_depth('sample_image2.jpg')
    
    # Check that the two depth maps are not identical
    assert not np.array_equal(depth_map, depth_map2), 'Different images should produce different depth maps'

test_estimate_depth()