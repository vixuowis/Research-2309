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

    # Load YOLOv5 model for detecting vehicles on roads
    model = yolov5.load("vehicles")

    # Detect vehicles in the image
    results = model(image_path)
    
    # Filter the detections by "Car" class and return the bounding boxes, scores, and classes of detected vehicles as a dictionary
    return dict({
        "boxes": [results.xyxy[int(i)] for i in list(results.pandas().name[results.pandas().name == "car"].index)],
        "scores": [float(i) for i in list(results.pandas().category[results.pandas().name == "car"])],
        "classes": ["car" for _ in range(len([_ for _ in results.pandas().name if _ == "car"]))]})

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