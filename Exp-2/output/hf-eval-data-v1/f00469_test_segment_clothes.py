def test_segment_clothes():
    # Test image URL
    test_image_url = 'https://example.com/test_image.jpg'
    # Call the function with the test image URL
    result = segment_clothes(test_image_url)
    # Check the result is not None
    assert result is not None
    # Check the result is a tensor
    assert isinstance(result, torch.Tensor)

test_segment_clothes()