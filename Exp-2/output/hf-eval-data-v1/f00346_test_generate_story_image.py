def test_generate_story_image():
    '''
    This function tests the generate_story_image function.
    '''
    # Define a test prompt
    test_prompt = 'a scene of a magical forest with fairies and elves'
    
    # Generate an image
    filename = generate_story_image(test_prompt)
    
    # Load the image
    image = Image.open(filename)
    
    # Check that an image was generated
    assert image is not None, 'No image was generated.'
    
    # Check that the image has the correct dimensions
    assert image.size == (512, 512), 'The image has the wrong dimensions.'
    
    # Check that the image is in the correct mode
    assert image.mode == 'RGB', 'The image is in the wrong mode.'
    
    # Delete the test image
    os.remove(filename)
    
    print('All tests passed.')

test_generate_story_image()