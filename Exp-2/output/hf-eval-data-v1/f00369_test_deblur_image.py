def test_deblur_image():
    """
    This function tests the deblur_image function by using a sample image.
    It asserts that the output is not None and that the output is an instance of the Image class.
    """
    # Use a sample image for testing
    sample_image_path = 'path/to/sample_image.jpg'
    
    # Call the deblur_image function
    output = deblur_image(sample_image_path)
    
    # Assert that the output is not None
    assert output is not None, 'Output is None'
    
    # Assert that the output is an instance of the Image class
    assert isinstance(output, Image.Image), 'Output is not an image'
    
    print('All tests passed.')

test_deblur_image()