# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detect CS:GO players in an image using YOLOv8 object detection model.

    Args:
        image_path (str): Path to the image file where players need to be detected.

    Returns:
        list: Bounding boxes of detected players.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        Exception: If the prediction process fails.
    """
    import os
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    model = YOLO('keremberke/yolov8m-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    if not results:
        raise Exception(f"Prediction failed for image: {image_path}")

    return results[0].boxes

# test_function_code --------------------

def test_detect_csgo_players():
    print("Testing started.")
    
    # Test case 1: Valid image
    print("Testing case [1/2] started.")
    test_image_path = 'path_to_valid_test_image.jpg'
    result = detect_csgo_players(test_image_path)
    assert type(result) == list, f"Test case [1/2] failed: Expected result type list, got {type(result)}"

    # Test case 2: Invalid image path
    print("Testing case [2/2] started.")
    invalid_image_path = 'non_existent_image.jpg'
    try:
        _ = detect_csgo_players(invalid_image_path)
        assert False, "Test case [2/2] failed: FileNotFoundError was expected but not raised."
    except FileNotFoundError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_csgo_players()