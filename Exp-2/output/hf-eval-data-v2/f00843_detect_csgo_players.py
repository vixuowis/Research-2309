# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detects players in a live game of Counter-Strike: Global Offensive (CS:GO) using a pre-trained YOLO model.

    Args:
        image_path (str): The path to the game screen image.

    Returns:
        A rendered image with detected players' bounding boxes.
    """
    model = YOLO('keremberke/yolov8m-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    return render

# test_function_code --------------------

def test_detect_csgo_players():
    """
    Tests the detect_csgo_players function by using a sample image.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_csgo_players(image_path)
    assert result is not None, 'No result returned.'
    assert isinstance(result, type(render_result)), 'Result is not of expected type.'

# call_test_function_code --------------------

test_detect_csgo_players()