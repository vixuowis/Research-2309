# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detect blood cells in an image using the YOLO model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        render: The rendered image with detected blood cells.
    """

    # Load the image
    img = cv2.imread(image_path)

    model = YOLO()

    results = model.detect(img)
    render = render_result(results, img)
    
    return render


# test_function_code --------------------

def test_detect_blood_cells():
    """
    Test the detect_blood_cells function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = detect_blood_cells(image_path)
    assert render is not None, 'No render returned'
    assert isinstance(render, type(render)), 'Render is not the correct type'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_blood_cells()