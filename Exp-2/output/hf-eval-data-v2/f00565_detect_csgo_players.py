# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image):
    """
    Detect and locate players in the given image using YOLO object detection model.

    Args:
        image (str): URL or local path to the image.

    Returns:
        render: A matplotlib figure object with the detections rendered on the original image.

    Raises:
        ValueError: If the image path is not valid.
    """
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image)
    render = render_result(model=model, image=image, result=results[0])
    return render

# test_function_code --------------------

def test_detect_csgo_players():
    """
    Test the detect_csgo_players function with a sample image.
    """
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = detect_csgo_players(image)
    assert isinstance(render, type(matplotlib.figure.Figure())), 'The return type is not correct.'

# call_test_function_code --------------------

test_detect_csgo_players()