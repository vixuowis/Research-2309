# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detect CS:GO players in the provided image.

    Args:
        image_path (str): URL or local path to the image

    Returns:
        tuple: returns (detected_players, rendered_image) where:
            - detected_players is a list of detected player bounding boxes
            - rendered_image is the visualization of the detections
    """
    # Create a YOLO object detection model
    model = YOLO('keremberke/yolov8n-csgo-player-detection')

    # Set the model parameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Predict the player positions
    results = model.predict(image_path)

    # Extract the player bounding boxes
    detected_players = results[0].boxes

    # Render the players' detections on the original image
    render = render_result(model=model, image=image_path, result=results[0])

    # Return the player bounding boxes and the rendered image
    return detected_players, render

# test_function_code --------------------

def test_detect_csgo_players():
    print("Testing detect_csgo_players function...")
    sample_image_path = 'test_image.jpg'  # Replace with a valid image path or URL

    # Test case 1: Detect players in an image
    print("Testing case 1...")
    detected_players, render = detect_csgo_players(sample_image_path)
    assert len(detected_players) >= 1, "Test case 1 failed: No players detected"

    # Test case 2: Verify the type of the returned result
    print("Testing case 2...")
    assert isinstance(detected_players, list), "Test case 2 failed: detected_players is not a list"
    assert 'show' in dir(render), "Test case 2 failed: render does not have a show method"

    print("All test cases for detect_csgo_players passed.")

# Run the tests
test_detect_csgo_players()