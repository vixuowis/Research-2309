# Test function for detect_potholes
# @param: None
# @return: None
def test_detect_potholes():
    # Test image URL
    test_image_url = 'https://github.com/ultralytics/yolov8/raw/master/data/images/zidane.jpg'
    # Run the function with the test image
    boxes, masks = detect_potholes(test_image_url)
    # Check if the function returns results
    assert boxes is not None and masks is not None, 'No results returned'
    # Check if the function returns correct types
    assert isinstance(boxes, list) and isinstance(masks, list), 'Incorrect result types'
    # Check if the function returns results of correct length
    assert len(boxes) > 0 and len(masks) > 0, 'Empty results returned'
    print('All tests passed.')

# Run the test function
test_detect_potholes()