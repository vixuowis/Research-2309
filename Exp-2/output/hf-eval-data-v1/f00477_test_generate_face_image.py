def test_generate_face_image():
    """
    This function tests the 'generate_face_image' function by generating an image and checking that the file is created.
    """
    # Define the model ID and save path
    model_id = 'google/ncsnpp-celebahq-256'
    save_path = 'test_generated_face.png'

    # Call the function to generate an image
    generate_face_image(model_id, save_path)

    # Check that the file was created
    assert os.path.exists(save_path), 'The image file was not created.'

    # Load the image file
    image = Image.open(save_path)

    # Check that the image is not empty
    assert np.array(image).any(), 'The generated image is empty.'