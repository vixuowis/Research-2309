# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_unknown_person(image):
    """
    Detects unknown person in the given image using YOLO object detection model.

    Args:
        image (str): The URL of the image to be processed.

    Returns:
        list: A list of bounding boxes for detected objects. Each bounding box is represented as a list of four numbers.
    """
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image)
    return results[0].boxes

# test_function_code --------------------

def test_detect_unknown_person():
    """
    Tests the detect_unknown_person function with different test cases.
    """
    image1 = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    image2 = 'https://placekitten.com/200/300'
    assert len(detect_unknown_person(image1)) > 0, 'Test Case 1 Failed'
    assert len(detect_unknown_person(image2)) == 0, 'Test Case 2 Failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_unknown_person()