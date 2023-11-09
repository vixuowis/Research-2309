# Test function for detect_objects_in_valorant
# The function uses a test image from the Valorant game
# The function asserts that the returned result is not None

def test_detect_objects_in_valorant():
    test_image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_objects_in_valorant(test_image)
    assert result is not None, 'No objects detected'
    print('Test passed')

test_detect_objects_in_valorant()