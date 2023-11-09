import numpy as np

# Test function for estimate_depth
# Uses a random drone footage for testing

def test_estimate_depth():
    # Generate a random drone footage
    drone_footage = np.random.rand(224, 224, 3)

    # Call the estimate_depth function
    depth_map = estimate_depth(drone_footage)

    # Check the shape of the depth map
    # The shape should be the same as the input footage
    assert depth_map.shape == drone_footage.shape, 'The shape of the depth map is not correct'

    # Check the values of the depth map
    # The values should be between 0 and 1
    assert np.all((depth_map >= 0) & (depth_map <= 1)), 'The values of the depth map are not correct'

# Call the test function
test_estimate_depth()