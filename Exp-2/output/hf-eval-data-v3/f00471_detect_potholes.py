# function_import --------------------

from ultralyticsplus import YOLO
from PIL import Image

# function_code --------------------

def detect_potholes(image_path):
    """
    Detects potholes in the given image using a pre-trained YOLO model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the bounding boxes and masks of the detected potholes.
    """
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    image = Image.open(image_path)
    results = model.predict(image)
    return {'boxes': results[0].boxes, 'masks': results[0].masks}

# test_function_code --------------------

def test_detect_potholes():
    """
    Tests the detect_potholes function with a sample image.
    """
    result = detect_potholes('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg')
    assert isinstance(result, dict)
    assert 'boxes' in result
    assert 'masks' in result
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_potholes()