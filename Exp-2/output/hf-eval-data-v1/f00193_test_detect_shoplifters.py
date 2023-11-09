def test_detect_shoplifters():
    """
    This function tests the detect_shoplifters function by passing a sample image and checking the output.
    """
    # Define the path to a sample image
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    # Call the detect_shoplifters function
    result = detect_shoplifters(image_path)
    # Check the output
    assert isinstance(result, dict), 'The output should be a dictionary.'
    assert 'boxes' in result, 'The output dictionary should contain bounding boxes.'
    assert 'scores' in result, 'The output dictionary should contain scores.'
    assert 'categories' in result, 'The output dictionary should contain categories.'
    assert len(result['boxes']) == len(result['scores']) == len(result['categories']), 'The lengths of boxes, scores, and categories should be equal.'

# Run the test function
test_detect_shoplifters()