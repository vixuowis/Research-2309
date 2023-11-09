def test_classify_product_image():
    """
    This function tests the 'classify_product_image' function by classifying a sample image.
    """
    # URL of a sample product image
    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    
    # Classify the sample image
    output = classify_product_image(url)
    
    # Check if the output is a torch.Tensor
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor.'
    
    # Check if the output tensor has the correct shape
    assert output.shape == (1, 1000), 'Output tensor should have shape (1, 1000).'

test_classify_product_image()