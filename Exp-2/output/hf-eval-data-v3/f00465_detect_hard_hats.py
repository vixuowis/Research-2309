# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_hard_hats(image_path: str, model_name: str = 'keremberke/yolov8m-hard-hat-detection', conf: float = 0.25, iou: float = 0.45, agnostic_nms: bool = False, max_det: int = 1000):
    """
    Detects hard hats in a given image using a pre-trained YOLO model.

    Args:
        image_path (str): The path to the image file.
        model_name (str, optional): The name of the pre-trained model. Defaults to 'keremberke/yolov8m-hard-hat-detection'.
        conf (float, optional): The confidence threshold for the model. Defaults to 0.25.
        iou (float, optional): The Intersection over Union (IoU) threshold for the model. Defaults to 0.45.
        agnostic_nms (bool, optional): If True, the model will be agnostic to the class of the detected objects. Defaults to False.
        max_det (int, optional): The maximum number of detections the model can make. Defaults to 1000.

    Returns:
        A tuple containing the detected boxes and the rendered image.
    """
    model = YOLO(model_name)
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return results[0].boxes, render

# test_function_code --------------------

def test_detect_hard_hats():
    """Tests the detect_hard_hats function."""
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes, render = detect_hard_hats(image_path)
    assert isinstance(boxes, list), 'The returned boxes should be a list.'
    assert len(boxes) > 0, 'At least one box should be detected.'
    assert isinstance(render, type(None)), 'The render should be an instance of NoneType.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_hard_hats()