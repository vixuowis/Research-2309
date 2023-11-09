def test_generate_image_from_text():
    '''
    This function tests the generate_image_from_text function.
    '''
    # Define a test prompt
    prompt = 'A beautiful landscape with a waterfall and a sunset'
    
    # Call the function with the test prompt
    generate_image_from_text(prompt)
    
    # Load the generated image
    generated_image = torch.load('./generated_image.png')
    
    # Assert that the generated image is not None
    assert generated_image is not None, 'The generated image should not be None.'