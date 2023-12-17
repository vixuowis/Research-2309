# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detects Counter-Strike: Global Offensive (CS:GO) players in an image.

    Args:
        image_path (str): The path to the image where players are to be detected.

    Returns:
        list: A list of bounding boxes for the detected players.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        Exception: If the model fails to perform detection.
    """
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    if not results:
        raise Exception('Model failed to detect any players.')

    detected_players = results[0].boxes
    return detected_players

# test_function_code --------------------

def test_detect_csgo_players():
    print("Testing started.")
    image_path = 'path/to/sample/csgo_image.jpg'  # Use a valid image path for testing

    # Testing case 1: Valid image path
    print("Testing case [1/3] started.")
    try:
        players_detected = detect_csgo_players(image_path)
        assert isinstance(players_detected, list), f"Test case [1/3] failed: Expected a list, got {type(players_detected)}"
    except FileNotFoundError:
        print("Image file not found. Please ensure the image_path is correct.")

    # Testing case 2: Invalid image path
    print("Testing case [2/3] started.")
    invalid_path = 'path/to/nonexistent/image.jpg'
    try:
        detect_csgo_players(invalid_path)
        assert False, "Test case [2/3] failed: FileNotFoundError not raised."
    except FileNotFoundError:
        pass

    # Testing case 3: Empty model results
    print("Testing case [3/3] started.")
    try:
        detect_csgo_players(image_path)
        # Assuming this path will cause the model to return no results
        assert False, "Test case [3/3] failed: Expected an exception for no model results."
    except Exception as e:
        if not str(e).startswith('Model failed to detect any players.'):
            raise
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_csgo_players()