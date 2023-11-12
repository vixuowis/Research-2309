# function_import --------------------

import yolov5

# function_code --------------------

def detect_vehicles(image_path):
    """
    Detect vehicles in the given image using YOLOv5 object detection model.

    Args:
        image_path (str): The path or URL to the image.

    Returns:
        dict: A dictionary containing the bounding boxes, scores, and categories of the detected vehicles.
    """
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    results = model(image_path, size=640, augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    return {'boxes': boxes, 'scores': scores, 'categories': categories}

# test_function_code --------------------

def test_detect_vehicles():
    """
    Test the detect_vehicles function.
    """
    image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_vehicles(image_url)
    assert isinstance(result, dict)
    assert 'boxes' in result
    assert 'scores' in result
    assert 'categories' in result
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_vehicles()