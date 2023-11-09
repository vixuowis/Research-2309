def test_generate_image_from_text():
    '''
    This function tests the generate_image_from_text function.
    It uses a sample text prompt and a sample control image, and checks if the output image is successfully generated.
    '''
    import os

    # Define a sample text prompt and a sample control image path
    prompt = 'royal chamber with fancy bed'
    control_image_path = './images/control.png'

    # Call the function with the sample inputs
    output_image_path = generate_image_from_text(prompt, control_image_path)

    # Check if the output image is successfully generated
    assert os.path.exists(output_image_path), 'The output image was not generated.'

    print('The test passed successfully.')

test_generate_image_from_text()