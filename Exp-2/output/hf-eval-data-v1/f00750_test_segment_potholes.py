def test_segment_potholes():
    """
    Test function for segment_potholes function.
    """
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = segment_potholes(image)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'boxes' in result, 'The result should contain bounding boxes.'
    assert 'masks' in result, 'The result should contain masks.'