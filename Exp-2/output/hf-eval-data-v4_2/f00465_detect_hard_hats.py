# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_hard_hats(image_path):
    """
    Detects hard hats in the given image using a pre-trained YOLOv8 model.

    Args:
        image_path (str): URL or local path to the image where detection is to be performed.

    Returns:
        tuple: A tuple containing detection boxes and the rendered image.

    Raises:
        ValueError: If the image_path is not a valid URL or local path.
    """
    model = YOLO('keremberke/yolov8m-hard-hat-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)
    if not results:
        raise ValueError('No hard hats detected or invalid image path.')
    render = render_result(model=model, image=image_path, result=results[0])
    return results[0].boxes, render

# test_function_code --------------------

def test_detect_hard_hats():
    print("Testing started.")
    sample_image_path = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"  # Replace with a valid image URL during actual testing

    # Test case 1: Check if function returns a tuple
    print("Testing case [1/3] started.")
    result = detect_hard_hats(sample_image_path)
    assert isinstance(result, tuple), f"Test case [1/3] failed: Function should return a tuple."

    # Test case 2: Check if the first element in the tuple is a list (detection boxes)
    print("Testing case [2/3] started.")
    assert isinstance(result[0], list), f"Test case [2/3] failed: First element of the tuple should be a list of detection boxes."

    # Test case 3: Check if detection is successful (list is not empty)
    print("Testing case [3/3] started.")
    assert result[0], f"Test case [3/3] failed: Detection list should not be empty."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_hard_hats()