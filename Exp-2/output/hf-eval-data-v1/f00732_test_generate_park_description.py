def test_generate_park_description():
    '''
    This function tests the generate_park_description function.
    It uses a sample image URL for testing.
    '''
    # Define a sample image URL
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    
    # Generate a description of the image
    description = generate_park_description(img_url)
    
    # Assert that the description is not empty
    assert description != '', 'The generated description is empty.'
    
    # Assert that the description is a string
    assert isinstance(description, str), 'The generated description is not a string.'

test_generate_park_description()