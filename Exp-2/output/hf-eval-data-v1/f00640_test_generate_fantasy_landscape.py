def test_generate_fantasy_landscape():
    '''
    This function tests the generate_fantasy_landscape function.
    '''
    # Define a test prompt
    test_prompt = 'a peaceful scene in a lush green forest with a crystal-clear river flowing through it, under a blue sky with fluffy white clouds'
    
    # Generate an image using the test prompt
    image_path = generate_fantasy_landscape(test_prompt)
    
    # Check that the image file exists
    assert os.path.exists(image_path), 'The image file does not exist.'
    
    # Check that the image file is not empty
    assert os.path.getsize(image_path) > 0, 'The image file is empty.'
    
    print('All tests passed.')

test_generate_fantasy_landscape()