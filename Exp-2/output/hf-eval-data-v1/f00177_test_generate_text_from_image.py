def test_generate_text_from_image():
    """
    This function tests the 'generate_text_from_image' function.
    It uses a sample image file and checks if the output is a string.
    """
    # Define a sample image path
    sample_image_path = 'path_to_sample_image.jpg'
    
    # Generate a text description based on the content of the sample image
    text_output = generate_text_from_image(sample_image_path)
    
    # Check if the output is a string
    assert isinstance(text_output, str), 'The output should be a string.'
    
    # Print a success message
    print('The test passed successfully.')

test_generate_text_from_image()