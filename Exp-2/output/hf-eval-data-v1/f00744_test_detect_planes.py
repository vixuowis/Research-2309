def test_detect_planes():
    # Test the detect_planes function with a sample image
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = detect_planes(image_path)
    # Check that the function returns a result
    assert results is not None
    # Check that the result contains boxes
    assert 'boxes' in results[0]
    # Check that the boxes are not empty
    assert len(results[0]['boxes']) > 0