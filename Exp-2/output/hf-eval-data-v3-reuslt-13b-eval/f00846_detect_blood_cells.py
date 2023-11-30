# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detects blood cells in a given image using a pre-trained YOLO model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        render: A render object containing the detection results.
    """

    yolo = YOLO()
    result = yolo(image_path)
    
    return render_result(yolo, result)

# test_function_code --------------------

def test_detect_blood_cells():
    """
    Tests the detect_blood_cells function with a sample image.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = detect_blood_cells(image_path)
    assert render is not None, 'No detection results'
    print('All Tests Passed')


# call_test_function_code --------------------

test_detect_blood_cells()