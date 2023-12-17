# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def predict_pothole_segmentation(image_path: str) -> dict:
    """
    Predict and segment potholes in the given image.

    Args:
        image_path (str): The URL or local path to the image containing potholes.

    Returns:
        dict: A dictionary containing the boxes and masks of detected potholes.

    Raises:
        ValueError: If the image_path is None or an empty string.
        FileNotFoundError: If the image_path does not exist.
    """
    if not image_path:
        raise ValueError('The image path cannot be None or empty.')
    # Load the trained YOLOv8 model for pothole segmentation
    model = YOLO('keremberke/yolov8m-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Predict and segment potholes
    results = model.predict(image_path)

    # Extract boxes and masks
    boxes = results[0].boxes if results[0].boxes is not None else []
    masks = results[0].masks if results[0].masks is not None else []
    return {'boxes': boxes, 'masks': masks}

# test_function_code --------------------

def test_predict_pothole_segmentation():
    print("Testing started.")
    # We do not have a real dataset function so this is just a placeholder.
    # In a real scenario, we would load a sample image from a dataset
    sample_image_path = 'https://example.com/image_with_potholes.jpg'

    # Test case 1: Valid image path
    print("Testing case [1/3] started.")
    result = predict_pothole_segmentation(sample_image_path)
    assert 'boxes' in result and 'masks' in result, f"Test case [1/3] failed: result does not have required keys."

    # Test case 2: Empty image path
    print("Testing case [2/3] started.")
    try:
        predict_pothole_segmentation('')
        assert False, f"Test case [2/3] failed: ValueError not raised for empty image path."
    except ValueError:
        pass

    # Test case 3: None image path
    print("Testing case [3/3] started.")
    try:
        predict_pothole_segmentation(None)
        assert False, f"Test case [3/3] failed: ValueError not raised for None image path."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_predict_pothole_segmentation()