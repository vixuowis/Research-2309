def test_generate_caption():
    """
    This function tests the 'generate_caption' function.
    """
    # Define the path to the test image
    test_image_path = 'path_to_test_image.jpg'
    
    # Generate the caption for the test image
    test_caption = generate_caption(test_image_path)
    
    # Assert that the caption is a string
    assert isinstance(test_caption, str), 'The caption must be a string.'
    
    # Assert that the caption is not empty
    assert test_caption != '', 'The caption cannot be empty.'

test_generate_caption()