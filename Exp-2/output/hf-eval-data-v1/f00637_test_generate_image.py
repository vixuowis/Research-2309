def test_generate_image():
    """
    This function tests the 'generate_image' function.
    """
    # Define a test prompt
    test_prompt = 'a lighthouse on a foggy island'
    
    # Define the save path for the test image
    test_save_path = 'test_image.png'
    
    # Call the function with the test prompt
    generate_image(test_prompt, test_save_path)
    
    # Check if the image file was created
    assert os.path.isfile(test_save_path), 'The image file was not created.'
    
    # If the file was created, delete it
    if os.path.isfile(test_save_path):
        os.remove(test_save_path)