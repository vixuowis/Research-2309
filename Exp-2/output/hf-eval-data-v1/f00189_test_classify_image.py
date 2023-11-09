def test_classify_image():
    """
    This function tests the classify_image function by classifying an image from a URL and checking the output.
    """
    # Define a URL of an image to be classified
    img_url = 'https://example.com/image.jpg'
    
    # Classify the image
    output = classify_image(img_url)
    
    # Check that the output is a torch.Tensor
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor'
    
    # Check that the output has the correct shape
    assert output.shape == (1, 1000), 'Output should have shape (1, 1000)'

test_classify_image()