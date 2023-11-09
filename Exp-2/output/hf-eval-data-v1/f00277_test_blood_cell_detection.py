def test_blood_cell_detection():
    # Test the blood_cell_detection function with a sample image
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = blood_cell_detection(image)
    # Assert that the function returns a result (does not return None)
    assert results is not None
    # Assert that the function correctly identifies at least one object in the image
    assert len(results[0].boxes) > 0

test_blood_cell_detection()