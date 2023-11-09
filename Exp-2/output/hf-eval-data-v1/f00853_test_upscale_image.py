def test_upscale_image():
    """
    Test function for the 'upscale_image' function.
    
    The function generates a random low-resolution image tensor, upscales it using the 'upscale_image' function,
    and asserts that the size of the upscaled image is twice the size of the original image.
    """
    # Generate a random low-resolution image tensor
    low_resolution_tensor = torch.rand((1, 3, 64, 64))
    
    # Upscale the image
    upscaled_tensor = upscale_image(low_resolution_tensor)
    
    # Assert that the size of the upscaled image is twice the size of the original image
    assert upscaled_tensor.shape == (1, 3, 128, 128), 'The upscaled image size is not twice the size of the original image.'

test_upscale_image()