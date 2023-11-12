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
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return render

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