def test_segment_image():
    """
    This function tests the 'segment_image' function with a sample image URL.
    """
    # Define a sample image URL for testing
    url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'
    
    # Call the 'segment_image' function with the sample image URL
    segmented_image = segment_image(url)
    
    # Assert that the function returns an instance of the PIL.Image class
    assert isinstance(segmented_image, Image.Image), 'The function should return a PIL.Image object.'

test_segment_image()