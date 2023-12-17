# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detect the location of players in an image from a CS:GO game using YOLO model.

    Parameters:
        image_path (str): The path to the image file to be processed.

    Returns:
        list: The detected players' bounding boxes.
    """
    # Initialize the YOLO model
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    # Set model overrides
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Process the image
    results = model.predict(image_path)
    # Extract detected players' locations
    detected_players = results[0].boxes
    return detected_players

# test_function_code --------------------

def test_detect_csgo_players():
    print("Testing detect_csgo_players function.")
    # Assume 'test_image.jpg' is a valid image path with players
    test_image_path = 'test_image.jpg'

    detected_players = detect_csgo_players(test_image_path)

    # Since we cannot know the exact output, we check for the type and length of the output
    assert isinstance(detected_players, list), "The function should return a list."
    print("Test case [1/1]: Success")
    print("Testing finished.")

test_detect_csgo_players()