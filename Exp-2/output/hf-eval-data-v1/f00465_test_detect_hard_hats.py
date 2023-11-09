def test_detect_hard_hats():
    # Test image path
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    # Call the function with the test image
    results = detect_hard_hats(image_path)
    # Assert that the results are not None
    assert results is not None
    # Assert that the results contain bounding boxes
    assert len(results[0].boxes) > 0

test_detect_hard_hats()