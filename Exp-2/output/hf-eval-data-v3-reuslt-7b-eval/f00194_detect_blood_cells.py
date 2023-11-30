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
    # Load and configure model
    yolo = YOLO("yolov5s")  # or yolov5m
    yolo.conf_thres, yolo.iou_thres = 0.25, 0.45
    
    # Run inference with image and return results
    img = yolo.predict(image_path)
    render = render_result(yolo, np.array(img), show=False)[0]  # or save to file instead of returning: `save_file=True`
    
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