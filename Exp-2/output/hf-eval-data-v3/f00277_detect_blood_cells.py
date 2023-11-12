# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detect and count platelets, red blood cells, and white blood cells in a digital blood sample image.

    Args:
        image_path (str): The path or URL of the blood sample image.

    Returns:
        dict: A dictionary containing the bounding boxes and class names of the detected blood cells.
    """
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()

    return results[0].boxes

# test_function_code --------------------

def test_detect_blood_cells():
    """
    Test the detect_blood_cells function with a sample blood image.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = detect_blood_cells(image_path)
    assert isinstance(results, list), 'The result should be a list of bounding boxes.'
    assert len(results) > 0, 'At least one blood cell should be detected.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_blood_cells()