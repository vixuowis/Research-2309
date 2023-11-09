# function_import --------------------

import yolov5

# function_code --------------------

def detect_objects(img_path):
    """
    Detect objects in an image using the YOLOv5 model.

    Args:
        img_path (str): The path to the image file.

    Returns:
        A tuple (boxes, scores, categories), where:
            - boxes (list): A list of bounding boxes for detected objects.
            - scores (list): A list of confidence scores for each detected object.
            - categories (list): A list of categories for each detected object.
    """
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    results = model(img_path, size=640, augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    return boxes, scores, categories

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    img_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes, scores, categories = detect_objects(img_path)
    assert len(boxes) > 0, 'No objects detected.'
    assert len(scores) > 0, 'No scores detected.'
    assert len(categories) > 0, 'No categories detected.'

# call_test_function_code --------------------

test_detect_objects()