def test_detect_objects():
    '''
    This function tests the detect_objects function with a sample image from the COCO 2017 dataset.
    '''
    # Define the URL of the sample image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Call the detect_objects function
    outputs = detect_objects(url)
    
    # Assert that the function returns a dictionary
    assert isinstance(outputs, dict)
    
    # Assert that the dictionary contains the expected keys
    assert 'logits' in outputs and 'pred_boxes' in outputs

test_detect_objects()