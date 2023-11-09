def test_generate_image():
    '''
    This function tests the generate_image function.
    It asserts that the function runs without errors and that it creates an image file.
    
    Parameters:
    None
    
    Returns:
    None
    '''
    import os
    
    # Call the function to test
    generate_image()
    
    # Assert that the image file was created
    assert os.path.exists('ddpm_generated_image.png')