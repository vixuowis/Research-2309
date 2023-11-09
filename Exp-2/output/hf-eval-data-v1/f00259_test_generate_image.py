def test_generate_image():
    """
    This function tests the generate_image function by generating an image based on a test prompt.
    
    Parameters:
    None
    
    Returns:
    None
    """
    # Define the test prompt
    test_prompt = 'A futuristic city under the ocean'
    
    # Call the function with the test prompt
    generate_image(test_prompt)
    
    # Check if the image file has been created
    assert os.path.exists(f'{test_prompt.replace(" ", "_")}.png')