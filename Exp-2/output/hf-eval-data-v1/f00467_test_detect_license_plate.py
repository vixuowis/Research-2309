def test_detect_license_plate():
    # Test the function with a test image
    img_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_license_plate(img_path)
    # Assert the result is not None
    assert result is not None
    # Assert the result is either 'Access granted' or 'Access denied'
    assert result in ['Access granted', 'Access denied']

test_detect_license_plate()