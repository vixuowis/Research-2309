def test_generate_image_from_text():
    """
    This function tests the 'generate_image_from_text' function by providing a text prompt and checking if an image file is created.
    """
    import os

    # Define the text prompt and the output file name
    prompt = 'A modern living room with a fireplace and a large window overlooking a forest.'
    output_file = 'test_image.png'

    # Call the function to generate the image
    generate_image_from_text(prompt, output_file)

    # Check if the image file is created
    assert os.path.isfile(output_file), 'The image file was not created.'

    # If the file is created, delete it after the test
    if os.path.isfile(output_file):
        os.remove(output_file)