# Test function for get_depth_information
# This function loads a test dataset, selects a sample image, and then uses the get_depth_information function to predict the depth information
# The predicted depth information is then compared with the actual depth information using an assert statement
# Note: The comparison is not strict due to the inherent variability in depth estimation

def test_get_depth_information():
    # Load the test dataset
    test_dataset = load_dataset('diode-subset')
    # Select a sample image
    sample_image = test_dataset[0]['image']
    # Get the actual depth information
    actual_depth_information = test_dataset[0]['depth']
    # Use the get_depth_information function to predict the depth information
    predicted_depth_information = get_depth_information(sample_image)
    # Compare the predicted and actual depth information
    assert abs(predicted_depth_information - actual_depth_information) < 0.1