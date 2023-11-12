# function_import --------------------

import yolov5

# function_code --------------------

def detect_shoplifters(image_path: str) -> dict:
    '''
    Detect potential shoplifters in the given image using the pre-trained YOLOv5 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the detected objects' bounding boxes, scores, and categories.
    '''
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    results = model(image_path)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    return {'boxes': boxes, 'scores': scores, 'categories': categories}

# test_function_code --------------------

def test_detect_shoplifters():
    '''
    Test the detect_shoplifters function.
    '''
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = detect_shoplifters(image_path)
    assert isinstance(results, dict)
    assert 'boxes' in results
    assert 'scores' in results
    assert 'categories' in results
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_shoplifters()