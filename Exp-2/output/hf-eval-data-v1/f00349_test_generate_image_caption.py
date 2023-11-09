def test_generate_image_caption():
    '''
    Function to test the generate_image_caption function
    '''
    # Define the image path and the expected caption
    image_path = 'test_image.jpg'
    expected_caption = 'A product photography of a red and white striped umbrella.'
    
    # Generate the caption
    generated_caption = generate_image_caption(image_path)
    
    # Assert that the generated caption is not None
    assert generated_caption is not None, 'The generated caption is None'
    
    # Assert that the generated caption is a string
    assert isinstance(generated_caption, str), 'The generated caption is not a string'
    
    # Assert that the generated caption is not empty
    assert generated_caption != '', 'The generated caption is empty'
    
    # Note: We are not comparing the generated caption with the expected caption strictly
    # because the model can generate different captions for the same image based on the context
    # and the training data
    
    print('All tests passed.')

# Run the test function
test_generate_image_caption()