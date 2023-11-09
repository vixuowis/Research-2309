def test_generate_image():
    '''
    This function tests the generate_image function.
    '''
    # Define the prompt
    prompt = 'luxury living room with a fireplace'
    # Define the image path
    image_path = 'test_images/test_image.png'
    # Define the output path
    output_path = 'test_images/output_image.png'
    # Call the function
    generate_image(prompt, image_path, output_path)
    # Load the generated image
    generated_image = Image.open(output_path)
    # Assert that the image was generated
    assert generated_image is not None