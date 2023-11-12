# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_potholes(image_path: str) -> dict:
    '''
    Detects potholes in the given image using a pre-trained YOLOv8 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the bounding boxes and masks of the detected potholes.
    '''
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return {'boxes': results[0].boxes, 'masks': results[0].masks, 'render': render}

# test_function_code --------------------

def test_detect_potholes():
    '''
    Tests the detect_potholes function with a sample image.
    '''
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_potholes(image_path)
    assert isinstance(result, dict), 'Result should be a dictionary.'
    assert 'boxes' in result, 'Result should contain bounding boxes.'
    assert 'masks' in result, 'Result should contain masks.'
    assert 'render' in result, 'Result should contain a render of the detection.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_potholes()