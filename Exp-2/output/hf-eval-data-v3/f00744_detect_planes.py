# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_planes(image_path):
    '''
    Detects planes in the given image using the YOLO model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A list of bounding boxes for detected planes.
    '''
    model = YOLO('keremberke/yolov8m-plane-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    boxes = results[0].boxes
    return boxes

# test_function_code --------------------

def test_detect_planes():
    '''
    Tests the detect_planes function.
    '''
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes = detect_planes(image_path)
    assert isinstance(boxes, list), 'The result should be a list.'
    assert all(isinstance(box, tuple) and len(box) == 4 for box in boxes), 'Each box should be a tuple of four elements.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_planes()