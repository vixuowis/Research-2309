def test_detect_objects():
    """
    This function tests the detect_objects function.
    It uses a sample image from the COCO 2017 validation dataset.
    """
    # Define the test image URL
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    
    # Call the function with the test image
    result = detect_objects(test_image_url)
    
    # Check the result
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'logits' in result, 'The result should contain logits.'
    assert 'bboxes' in result, 'The result should contain bounding boxes.'
    
    # Print a success message
    print('Test passed.')

# Run the test function
test_detect_objects()