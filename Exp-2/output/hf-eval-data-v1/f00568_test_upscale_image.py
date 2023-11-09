def test_upscale_image():
    """
    Function to test the upscale_image function.
    
    The function loads a test image, upscales it using the upscale_image function, and then checks if the upscaled image exists.
    """
    # Define the path to the test image
    test_image_path = 'test_image.jpg'
    
    # Upscale the test image
    upscale_image(test_image_path)
    
    # Check if the upscaled image exists
    assert os.path.exists('upscaled_' + test_image_path), 'Upscaled image does not exist.'
    
    print('All tests passed.')

# Run the test function
test_upscale_image()