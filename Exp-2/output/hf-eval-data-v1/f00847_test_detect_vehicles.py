def test_detect_vehicles():
    """
    Test the detect_vehicles function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes, scores, categories = detect_vehicles(image_path)
    assert len(boxes) > 0, 'No vehicles detected.'
    assert len(scores) > 0, 'No confidence scores returned.'
    assert len(categories) > 0, 'No categories returned.'
    assert len(boxes) == len(scores) == len(categories), 'Mismatch in lengths of returned values.'

test_detect_vehicles()