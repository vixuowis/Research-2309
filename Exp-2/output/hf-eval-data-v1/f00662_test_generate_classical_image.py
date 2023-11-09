def test_generate_classical_image():
    """
    This function tests the generate_classical_image function.
    It asserts that the output of the function is not None, indicating that an image was generated.
    """
    # Call the function to generate a classical image
    generated_image = generate_classical_image()
    
    # Assert that an image was generated
    assert generated_image is not None, 'No image was generated.'
    
    print('Test passed.')

test_generate_classical_image()