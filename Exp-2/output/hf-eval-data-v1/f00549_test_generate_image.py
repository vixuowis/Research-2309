def test_generate_image():
    '''
    This function tests the generate_image function by providing a sample prompt and checking the type of the output.
    '''
    # Define a sample prompt
    prompt = 'kangaroo eating pizza'
    
    # Generate the image
    image = generate_image(prompt)
    
    # Check the type of the output
    assert isinstance(image, type(torch.zeros(1)))