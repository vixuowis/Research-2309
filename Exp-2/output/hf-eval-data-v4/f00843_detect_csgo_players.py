# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path):
    """
    Detect CS:GO players in the provided game image.

    Args:
    - image_path (str): The path to the game screen image.

    Returns:
    - list: The bounding boxes of the detected players.
    - object: The rendered image with bounding boxes.
    """
    # Initialize the YOLO model
    model = YOLO('keremberke/yolov8m-csgo-player-detection')
    # Set model parameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Predict and get results
    results = model.predict(image_path)
    print(results[0].boxes)
    # Render the results on the original image
    render = render_result(model=model, image=image_path, result=results[0])
    return results[0].boxes, render

# test_function_code --------------------

import os
from PIL import Image
from ultralyticsplus import YOLO

def test_detect_csgo_players():
    print("Testing started.")
    # Assume 'test_image.jpg' is an image from the CS:GO dataset for testing
    test_image = 'test_image.jpg'
    
    # Testing case 1: Detect players in the test image
    print("Testing case [1/1] started.")
    boxes, render = detect_csgo_players(test_image)
    assert isinstance(boxes, list), f"Test case [1/1] failed: bounding boxes not returned as a list"
    assert isinstance(render, Image.Image), f"Test case [1/1] failed: render is not an image object"
    
    print(f"Detected {len(boxes)} players.")
    print("Testing finished.")

# Run the test function
test_detect_csgo_players()