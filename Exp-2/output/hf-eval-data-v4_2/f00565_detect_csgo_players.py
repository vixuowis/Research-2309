# requirements_file --------------------

pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path: str):
    """
    Detect and locate Counter-Strike: Global Offensive players in the given image.

    Args:
        image_path (str): The URL or local path to the input image for detection.

    Returns:
        A results object containing detected players with their bounding boxes and a rendered image with the detections.
    """
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return results, render

# test_function_code --------------------

def test_detect_csgo_players():
    print("Testing started.")
    sample_image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'  # Sample image URL for testing

    # Test case 1: Check if the function returns two items
    print("Testing case [1/3] started.")
    results, render = detect_csgo_players(sample_image_path)
    assert len((results, render)) == 2, f"Test case [1/3] failed: Expected 2 return values, got {len((results, render))}"

    # Test case 2: Check for detection in results
    print("Testing case [2/3] started.")
    assert 'boxes' in results[0], f"Test case [2/3] failed: 'boxes' not found in results"

    # Test case 3: Check for a rendered result
    print("Testing case [3/3] started.")
    assert hasattr(render, 'show'), f"Test case [3/3] failed: Render object does not have a 'show' method"
    print("Testing finished.")

    return 'All test cases passed!'

# call_test_function_line --------------------

test_detect_csgo_players()