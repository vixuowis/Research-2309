def test_generate_butterfly_image():
    """
    This function tests the 'generate_butterfly_image' function by generating an image and checking if the output is not None.
    """
    # Generate a butterfly image
    generated_image = generate_butterfly_image()
    
    # Check if the generated image is not None
    assert generated_image is not None, 'The generated image should not be None.'
    
    print('All tests passed.')

test_generate_butterfly_image()