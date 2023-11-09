def test_generate_image_description():
    """
    This function tests the 'generate_image_description' function.
    It uses a sample text to generate an image description and checks if the output is not None.
    """
    # Define a sample text
    sample_text = 'A beautiful sunset over the ocean.'
    
    # Generate an image description based on the sample text
    result = generate_image_description(sample_text)
    
    # Check if the output is not None
    assert result is not None, 'The function did not return a result.'
    
    print('All tests passed.')

test_generate_image_description()