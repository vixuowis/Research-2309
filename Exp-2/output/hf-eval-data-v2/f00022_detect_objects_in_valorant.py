# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_objects_in_valorant(game_frame):
    """
    Detects objects in a Valorant game frame using a YOLO model.

    Args:
        game_frame (str): The image of the game frame.

    Returns:
        A render object that visualizes the detected objects in the game frame.
    """
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(game_frame)
    render = render_result(model=model, image=game_frame, result=results[0])
    return render

# test_function_code --------------------

def test_detect_objects_in_valorant():
    """
    Tests the detect_objects_in_valorant function by using a sample game frame.
    """
    game_frame = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = detect_objects_in_valorant(game_frame)
    assert render is not None, 'No objects detected.'

# call_test_function_code --------------------

test_detect_objects_in_valorant()