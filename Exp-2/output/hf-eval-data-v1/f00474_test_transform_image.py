def test_transform_image():
    # Test the transform_image function with a sample image
    transformed_image = transform_image('path/to/sample/image.jpg')
    # Assert that the function returns a torch.Tensor (the type returned by the Hugging Face pipeline)
    assert isinstance(transformed_image, torch.Tensor)
    # Assert that the function does not return an empty tensor
    assert transformed_image.nelement() > 0

test_transform_image()