def test_generate_anime_style_image():
    '''
    Function to test the 'generate_anime_style_image' function.
    
    Parameters:
    None
    
    Returns:
    None
    '''
    # Define a test prompt
    test_prompt = 'anime-style girl with a guitar'
    
    # Call the function with the test prompt
    generate_anime_style_image(test_prompt)
    
    # Check if the image file has been created
    assert os.path.exists('./anime_girl_guitar.png'), 'Image file not created.'