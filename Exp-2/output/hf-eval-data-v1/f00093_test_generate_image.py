def test_generate_image():
    '''
    This function tests the generate_image function by generating an image with a given text prompt and negative prompt.
    '''
    prompt = 'two tigers'
    negative_prompt = 'bad, deformed, ugly, bad anatomy'
    image = generate_image(prompt, negative_prompt)
    
    assert isinstance(image, Image.Image), 'The output should be an instance of PIL.Image.'
    
    # Save the image for manual inspection
    image.save('test_generated_image.png')

test_generate_image()