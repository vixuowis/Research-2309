# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_unknown_person(surveillance_image):
    """
    Detects unknown persons in a given surveillance image using a YOLOv8 model.

    Args:
        surveillance_image (str): The URL or local path of the surveillance image.

    Returns:
        A list of bounding boxes for detected objects. Each bounding box is represented as a list of four numbers,
        indicating the top-left and bottom-right corners of the box.
    """
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(surveillance_image)
    return results[0].boxes

# test_function_code --------------------

def test_detect_unknown_person():
    """
    Tests the detect_unknown_person function by using a sample image.
    """
    sample_image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes = detect_unknown_person(sample_image)
    assert isinstance(boxes, list), 'The result should be a list.'
    for box in boxes:
        assert len(box) == 4, 'Each bounding box should have four elements.'
        assert all(isinstance(corner, float) for corner in box), 'Each corner of the bounding box should be a float.'

# call_test_function_code --------------------

test_detect_unknown_person()