def test_segment_city_layout():
    # Test the function with a sample image
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    output = segment_city_layout(image_url)
    
    # Check that the output is not None
    assert output is not None
    
    # Check that the output has the correct shape
    # The output should have shape (1, num_classes, height, width)
    assert len(output.shape) == 4
    assert output.shape[0] == 1
    
    # Run the function with another sample image
    image_url = 'http://images.cocodataset.org/val2017/000000039770.jpg'
    output = segment_city_layout(image_url)
    
    # Check that the output is not None
    assert output is not None
    
    # Check that the output has the correct shape
    assert len(output.shape) == 4
    assert output.shape[0] == 1

test_segment_city_layout()