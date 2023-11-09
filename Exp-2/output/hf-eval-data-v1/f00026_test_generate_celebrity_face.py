def test_generate_celebrity_face():
    """
    This function tests the generate_celebrity_face function.
    It asserts that the function correctly generates an image and saves it to the given path.
    """
    # Define the model id and the save path
    model_id = 'google/ddpm-ema-celebahq-256'
    save_path = 'test_generated_celebrity_face.png'
    
    # Call the function
    generate_celebrity_face(model_id, save_path)
    
    # Assert that the image was correctly saved to the given path
    assert os.path.exists(save_path), 'The image was not saved to the given path.'
    
    # Remove the test image
    os.remove(save_path)