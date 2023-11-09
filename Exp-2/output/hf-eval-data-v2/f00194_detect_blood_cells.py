# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detect blood cells in an image using the YOLO model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A rendered image with detected blood cells.
    """
    model = YOLO('keremberke/yolov8n-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    detected_boxes = results[0].boxes
    render = render_result(model=model, image=image_path, result=results[0])
    return render

# test_function_code --------------------

def test_detect_blood_cells():
    """
    Test the detect_blood_cells function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_blood_cells(image_path)
    assert result is not None, 'No result returned.'
    assert isinstance(result, type(render_result)), 'Result is not of expected type.'

# call_test_function_code --------------------

test_detect_blood_cells()