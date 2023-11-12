# function_import --------------------

import yolov5

# function_code --------------------

def detect_objects(image_path):
    """
    Detect objects in an image using the YOLOv5 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the bounding boxes, scores, and categories of the detected objects.
    """
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    results = model(image_path, size=640, augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4].tolist()
    scores = predictions[:, 4].tolist()
    categories = predictions[:, 5].tolist()
    return {'boxes': boxes, 'scores': scores, 'categories': categories}

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = detect_objects(image_path)
    assert isinstance(results, dict)
    assert 'boxes' in results
    assert 'scores' in results
    assert 'categories' in results
    assert isinstance(results['boxes'], list)
    assert isinstance(results['scores'], list)
    assert isinstance(results['categories'], list)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_objects()