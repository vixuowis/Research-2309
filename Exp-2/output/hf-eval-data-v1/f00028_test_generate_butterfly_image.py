def test_generate_butterfly_image():
    """
    This function tests the 'generate_butterfly_image' function by generating an image and checking if the output is not None.
    """
    # Generate the butterfly image
    image = generate_butterfly_image()
    
    # Check if the image is not None
    assert image is not None, 'The generated image is None.'
    
    print('The test passed successfully.')

# Run the test function
test_generate_butterfly_image()