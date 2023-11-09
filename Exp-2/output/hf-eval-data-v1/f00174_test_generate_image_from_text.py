def test_generate_image_from_text():
    '''
    This function tests the generate_image_from_text function.
    '''
    # Define a test textual description
    test_text_description = 'A beautiful sunset over the ocean'
    
    # Generate the image from the test textual description
    test_generated_image = generate_image_from_text(test_text_description)
    
    # Assert that the generated image is not None
    assert test_generated_image is not None, 'The generated image should not be None.'
    
    # Assert that the generated image is of the correct type
    assert isinstance(test_generated_image, type(expected_output)), 'The generated image should be of the correct type.'
    
    # Assert that the generated image is not empty
    assert test_generated_image.size != 0, 'The generated image should not be empty.'
    
    # Call the test function
    test_generate_image_from_text()