# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detect and locate players in the given image using YOLO object detection model.

    Args:
        image_path (str): URL or local path to the image.

    Returns:
        render: A matplotlib figure object with the detections rendered on the original image.
    """
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return render

# test_function_code --------------------

def test_detect_csgo_players():
    """
    Test the detect_csgo_players function with a few test cases.
    """
    test_image1 = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    test_image2 = 'https://placekitten.com/200/300'
    assert isinstance(detect_csgo_players(test_image1), type(None))
    assert isinstance(detect_csgo_players(test_image2), type(None))
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_csgo_players()