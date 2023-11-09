def test_generate_insect_image():
    """
    This function tests the generate_insect_image function by generating an image and checking if the file exists.
    """
    # Define the model name and output path for the test
    model_name = 'schdoel/sd-class-AFHQ-32'
    output_path = 'test_insect_image.png'

    # Call the function to generate an insect image
    generate_insect_image(model_name, output_path)

    # Check if the image file exists at the output path
    assert os.path.exists(output_path), 'Image file does not exist'

    print('Test passed')

# Run the test function
test_generate_insect_image()