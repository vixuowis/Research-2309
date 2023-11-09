def test_upscale_image():
    '''
    This function tests the upscale_image function by providing a text prompt and checking the output.
    '''
    # Define a text prompt
    prompt = 'a photo of a movie character'
    
    # Call the upscale_image function
    upscaled_image = upscale_image(prompt)
    
    # Check the type of the output
    assert isinstance(upscaled_image, torch.Tensor), 'The output should be a torch.Tensor.'
    
    # Check the shape of the output
    assert len(upscaled_image.shape) == 3, 'The output should be a 3D tensor.'
    
    # Check the values of the output
    assert torch.min(upscaled_image) >= 0 and torch.max(upscaled_image) <= 1, 'The values should be between 0 and 1.'

test_upscale_image()