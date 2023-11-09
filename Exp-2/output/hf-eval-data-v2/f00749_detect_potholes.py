# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_potholes(image_path):
    """
    Detects potholes in an image using a pre-trained YOLOv8 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A rendered image with the detected potholes highlighted.
    """
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return render

# test_function_code --------------------

def test_detect_potholes():
    """
    Tests the detect_potholes function by passing a sample image and checking the output.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_potholes(image_path)
    assert result is not None, 'No result returned'
    assert isinstance(result, type(render_result)), 'Result is not of expected type'

# call_test_function_code --------------------

test_detect_potholes()