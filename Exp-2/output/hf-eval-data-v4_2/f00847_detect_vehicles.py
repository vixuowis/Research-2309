# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_vehicles(image_path):
    """
    Detect vehicles in the provided image using YOLOv5 model.

    Args:
        image_path (str): A path or URL to the traffic camera image.

    Returns:
        tuple: A tuple containing the predictions, boxes, scores, and categories for the detected vehicles.

    Raises:
        ValueError: If the image_path is not valid.
    """
    # Load the pre-trained YOLOv5 model
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    # Set detection parameters
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    # Verify the image_path
    if not os.path.exists(image_path) and not image_path.startswith('http://') and not image_path.startswith('https://'):
        raise ValueError('Invalid image path')
    # Perform object detection
    results = model(image_path, size=640, augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    return predictions, boxes, scores, categories

# test_function_code --------------------

def test_detect_vehicles():
    print("Testing started.")
    # Test case 1: Valid image URL
    print("Testing case [1/3] started.")
    predictions, boxes, scores, categories = detect_vehicles('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg')
    assert len(predictions) > 0, "Test case [1/3] failed: No predictions were made."
    # Test case 2: Non-existent file path
    print("Testing case [2/3] started.")
    try:
        detect_vehicles('non_existent_file.jpg')
    except ValueError as e:
        assert str(e) == 'Invalid image path', "Test case [2/3] failed: Wrong error message."
    # Test case 3: Invalid URL
    print("Testing case [3/3] started.")
    try:
        detect_vehicles('invalid_url')
    except ValueError as e:
        assert str(e) == 'Invalid image path', "Test case [3/3] failed: Wrong error message."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_vehicles()