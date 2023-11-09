def test_classify_pet_image():
    """
    This function tests the 'classify_pet_image' function by classifying a few sample images.
    """
    # Define the paths to the sample images
    sample_image_paths = ['path/to/sample1.jpg', 'path/to/sample2.jpg', 'path/to/sample3.jpg']
    
    # Define the expected class labels for the sample images
    expected_labels = ['cat', 'dog', 'cat']
    
    # Classify each sample image and compare the result with the expected label
    for i in range(len(sample_image_paths)):
        assert classify_pet_image(sample_image_paths[i]) == expected_labels[i]