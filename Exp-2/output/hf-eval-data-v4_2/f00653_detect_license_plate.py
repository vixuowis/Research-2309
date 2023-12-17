# requirements_file --------------------

pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_license_plate(image_path):
    """
    Detects license plates in the provided image using a pre-trained YOLOv5 model.

    Args:
        image_path (str): The file path or URL to the image for license plate detection.

    Returns:
        tuple: A tuple containing:
            - List of bounding boxes for detected license plates.
            - List of confidence scores for each detected license plate.
            - List of categories for each detected license plate.

    Raises:
        ValueError: If the provided image_path is not accessible or invalid.
    """
    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = 0.25  # Set confidence threshold
    model.iou = 0.45  # Set IoU threshold
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    results = model(image_path, size=640)  # Process the image
    predictions = results.pred[0]  # Get predictions

    if len(predictions) == 0:
        raise ValueError('No license plates detected.')

    boxes = predictions[:, :4].tolist()  # Extract bounding box coordinates
    scores = predictions[:, 4].tolist()  # Extract confidence scores
    categories = predictions[:, 5].tolist()  # Extract detected categories

    return boxes, scores, categories

# test_function_code --------------------

def test_detect_license_plate():
    from pathlib import Path

    print("Testing started.")
    # Assuming the images are stored in a directory called 'test_images'
    test_image_dir = Path('test_images')
    test_images = list(test_image_dir.glob('*.jpg'))

    # Test case 1: Verify that the function runs without error for a valid image path
    print("Testing case [1/3] started.")
    image_path = str(test_images[0])
    boxes, scores, categories = detect_license_plate(image_path)
    assert len(boxes) > 0, "Test case [1/3] failed: No boxes detected."

    # Test case 2: Verify that the function returns the correct data types
    print("Testing case [2/3] started.")
    assert isinstance(boxes, list), "Test case [2/3] failed: boxes is not a list."
    assert isinstance(scores, list), "Test case [2/3] failed: scores is not a list."
    assert isinstance(categories, list), "Test case [2/3] failed: categories is not a list."

    # Test case 3: Verify that the function raises an error for an invalid image path
    print("Testing case [3/3] started.")
    invalid_image_path = 'invalid/path/to/image.jpg'
    try:
        detect_license_plate(invalid_image_path)
        assert False, "Test case [3/3] passed: ValueError was not raised for an invalid image path."
    except ValueError as e:
        assert str(e) == 'No license plates detected.', "Test case [3/3] failed: Wrong error message was raised."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_license_plate()