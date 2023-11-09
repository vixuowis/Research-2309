# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_hard_hats(image_path):
    """
    Detects workers wearing hard hats in a given image using a pre-trained YOLO model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        A tuple containing the bounding boxes of detected hard hats and the rendered image.
    """
    model = YOLO('keremberke/yolov8m-hard-hat-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return results[0].boxes, render

# test_function_code --------------------

def test_detect_hard_hats():
    """
    Tests the detect_hard_hats function by checking if it returns the expected output on a test image.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes, render = detect_hard_hats(image_path)
    assert len(boxes) > 0, 'No hard hats detected.'
    assert render is not None, 'Rendered image is None.'

# call_test_function_code --------------------

test_detect_hard_hats()