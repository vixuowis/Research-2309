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
    model = yolov5.YoloV5()  # Load the model at first time only
    result = model(image_path)  # Detection
    return result['common']['boxes'][result['common']['labels'].tolist().index('car')], \
           result['common']['scores'], \
           [1] * len(result['common']['scores'])

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