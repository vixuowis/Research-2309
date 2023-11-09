def test_generate_image():
    '''
    This function tests the generate_image function by providing a sample prompt and checking if an image file is created.
    
    Parameters:
    None
    
    Returns:
    None
    '''
    import os
    # Define a sample prompt
    prompt = 'a serene lake at sunset'
    # Call the function with the sample prompt
    generate_image(prompt)
    # Check if the image file is created
    assert os.path.isfile('generated_image.png'), 'Image file not created'