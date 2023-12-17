# requirements_file --------------------

import subprocess

requirements = ["ultralyticsplus"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from ultralyticsplus import YOLO, Image

# function_code --------------------

def detect_potholes_in_image(image_path: str):
    """
    Detects potholes in a road image using a pre-trained YOLOv8 model.

    Args:
        image_path: A string specifying the URL or local path to the road image.

    Returns:
        A dictionary containing the bounding boxes and masks of detected potholes.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
        Exception: If any other error occurs during the detection process.
    """
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    try:
        image = Image.open(image_path)
    except FileNotFoundError as e:
        raise FileNotFoundError("Image file not found at specified path.") from e
    results = model.predict(image)
    return {'boxes': results[0].boxes, 'masks': results[0].masks}

# test_function_code --------------------

def test_detect_potholes_in_image():
    print("Testing started.")
    image_path = 'test_image.jpg'

    # Test case 1: Check if the function is detecting potholes correctly.
    print("Testing case [1/1] started.")
    try:
        detection_results = detect_potholes_in_image(image_path)
        if detection_results and \
           'boxes' in detection_results and \
           'masks' in detection_results:
            print("Test case [1/1] passed.")
        else:
            raise AssertionError("Test case [1/1] failed: No potholes detected.")
    except Exception as e:
        raise AssertionError(f"Test case [1/1] failed with error: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_potholes_in_image()