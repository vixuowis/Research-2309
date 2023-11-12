# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_players(image_path):
    """
    Detect the location of players in an image from a Counter-Strike: Global Offensive (CS:GO) game.

    Args:
        image_path (str): URL or local path to the image.

    Returns:
        list: Detected player locations.

    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    """
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    detected_players = results[0].boxes
    return detected_players

# test_function_code --------------------

def test_detect_players():
    """
    Test the detect_players function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    detected_players = detect_players(image_path)
    assert isinstance(detected_players, list), 'The result should be a list.'
    assert len(detected_players) > 0, 'There should be at least one player detected.'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_detect_players())