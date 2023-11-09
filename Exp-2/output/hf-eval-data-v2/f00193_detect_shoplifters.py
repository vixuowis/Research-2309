# function_import --------------------

import yolov5

# function_code --------------------

def detect_shoplifters(image_path):
    """
    Detect potential shoplifters in a given image using the YOLOv5 object detection model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A list of detected objects, each represented as a tuple of bounding box coordinates, score, and category.
    """
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
    return list(zip(boxes, scores, categories))

# test_function_code --------------------

def test_detect_shoplifters():
    """
    Test the detect_shoplifters function.
    """
    test_image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    detections = detect_shoplifters(test_image_path)
    assert isinstance(detections, list)
    for detection in detections:
        assert isinstance(detection, tuple)
        assert len(detection) == 3
        assert isinstance(detection[0], np.ndarray)
        assert isinstance(detection[1], float)
        assert isinstance(detection[2], float)

# call_test_function_code --------------------

test_detect_shoplifters()