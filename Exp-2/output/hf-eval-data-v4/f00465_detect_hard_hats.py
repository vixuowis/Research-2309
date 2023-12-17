# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_hard_hats(image_path):
    """Detect hard hats in a given image using YOLOv8 model.

    Args:
        image_path (str): URL or local path to the image.

    Returns:
        tuple: A tuple containing detected boxes and a rendered image.
    """
    model = YOLO('keremberke/yolov8m-hard-hat-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return results[0].boxes, render

# test_function_code --------------------

def test_detect_hard_hats():
    print("Testing detect_hard_hats function.")
    image_path = 'path_to_test_image.jpg'  # Replace with actual image path in test environment

    # Testing the function
    detected_boxes, render = detect_hard_hats(image_path)

    # Test case: Check if the function returns at least one detection
    print("Testing for at least one detection.")
    assert len(detected_boxes) > 0, "Test failed: No detections found."

    print("Testing for valid box dimensions.")
    for box in detected_boxes:
        assert len(box) == 4, "Test failed: Invalid box dimensions."  # Each box should have 4 values

    print("Testing for valid rendering.")
    assert render is not None, "Test failed: Render is None."

    print("All tests passed.")

# Run the test function
if __name__ == '__main__':
    test_detect_hard_hats()