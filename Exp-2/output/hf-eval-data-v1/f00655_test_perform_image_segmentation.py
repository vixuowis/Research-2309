def test_perform_image_segmentation():
    """
    This function tests the 'perform_image_segmentation' function by using a sample image.
    """
    # Load a sample image
    sample_image = load_sample_image()
    
    # Perform semantic segmentation on the sample image
    segmented_image = perform_image_segmentation(sample_image)
    
    # Check if the output is not None
    assert segmented_image is not None
    
    # Check if the output is an instance of Image
    assert isinstance(segmented_image, Image)

test_perform_image_segmentation()