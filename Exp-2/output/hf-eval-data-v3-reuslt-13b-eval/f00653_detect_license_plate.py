# function_import --------------------

import yolov5

# function_code --------------------

def detect_license_plate(img_path: str) -> dict:
    """
    Detects license plates in the given image using a pre-trained YOLOv5 model.

    Args:
        img_path (str): The path or URL to the image.

    Returns:
        dict: A dictionary containing the detected license plates' bounding boxes, scores, and categories.
    """
    
    # load yolov5 model
    model = yolov5.load("yolov5m6.pt")

    # set img size to 416x416 (YOLOv5's input size)
    model.img_size = 416
    
    # detect license plates in the given image
    result = model(img_path).pandas().xyxy[0]

    return dict(result)


# test_function_code --------------------

def test_detect_license_plate():
    """
    Tests the detect_license_plate function with a few test cases.
    """
    test_img1 = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    test_img2 = 'https://placekitten.com/200/300'

    result1 = detect_license_plate(test_img1)
    result2 = detect_license_plate(test_img2)

    assert isinstance(result1, dict), 'Result should be a dictionary.'
    assert isinstance(result2, dict), 'Result should be a dictionary.'
    assert 'boxes' in result1, 'Result dictionary should contain boxes.'
    assert 'scores' in result1, 'Result dictionary should contain scores.'
    assert 'categories' in result1, 'Result dictionary should contain categories.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_license_plate()