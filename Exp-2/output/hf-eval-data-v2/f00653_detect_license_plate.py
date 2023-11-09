# function_import --------------------

import yolov5

# function_code --------------------

def detect_license_plate(img_path):
    """
    Detects license plates in the given image using a pre-trained YOLOv5 model.

    Args:
        img_path (str): The path or URL to the image.

    Returns:
        A tuple (boxes, scores, categories), where:
            boxes (list): A list of bounding boxes for detected license plates.
            scores (list): A list of confidence scores for each detected license plate.
            categories (list): A list of categories for each detected license plate.
    """
    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    results = model(img_path, size=640)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    return boxes, scores, categories

# test_function_code --------------------

def test_detect_license_plate():
    """
    Tests the detect_license_plate function.
    """
    img_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes, scores, categories = detect_license_plate(img_path)

    assert len(boxes) > 0, 'No license plates detected.'
    assert len(scores) > 0, 'No confidence scores returned.'
    assert len(categories) > 0, 'No categories returned.'

# call_test_function_code --------------------

test_detect_license_plate()