def test_generate_image():
    '''
    This function tests the generate_image function by providing a sample prompt and checking if an image file is created.
    
    Parameters:
    None
    
    Returns:
    None
    '''
    # Define a sample prompt
    prompt = 'A vintage sports car racing through a desert landscape during sunset'
    
    # Call the generate_image function
    generate_image(prompt)
    
    # Check if the image file is created
    assert os.path.exists('./vintage_sports_car_desert_sunset.png'), 'Image file not created.'