# requirements_file --------------------

import subprocess

requirements = ["ultralyticsplus", "ultralytics"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path, conf_threshold=0.25, iou_threshold=0.45, agnostic_nms=False, max_det=1000):
    """Detect blood cells in an image.

    Args:
        image_path (str): The path or URL to the image.
        conf_threshold (float): The confidence threshold for detection.
        iou_threshold (float): The IoU threshold for detection.
        agnostic_nms (bool): Apply agnostic non-maximum suppression.
        max_det (int): The maximum number of detections.

    Returns:
        dict: Detected blood cells and their bounding boxes.

    Raises:
        ValueError: If image_path is incorrect or image can't be loaded.
    """
    model = YOLO('keremberke/yolov8n-blood-cell-detection')
    model.overrides['conf'] = conf_threshold
    model.overrides['iou'] = iou_threshold
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det
    results = model.predict(image_path)
    if results is None:
        raise ValueError('Unable to load image or incorrect path.')
    detected_cells = {'boxes': results[0].boxes, 'classes': results[0].pred_classes}
    render_image = render_result(model=model, image=image_path, result=results[0])
    return {'detected_cells': detected_cells, 'render_image': render_image}

# test_function_code --------------------

def test_detect_blood_cells():
    from io import BytesIO
    import requests
    from PIL import Image

    print("Testing started.")

    image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    image_path = BytesIO(requests.get(image_url).content)
    Image.open(image_path)  # Just to ensure that the image is loadable

    # Testing case 1: Check if function returns a dictionary.
    print("Testing case [1/3] started.")
    result_1 = detect_blood_cells(image_path)
    assert isinstance(result_1, dict), f"Test case [1/3] failed: Expected result to be a dict, got {type(result_1)}"

    # Testing case 2: Check if keys 'detected_cells' and 'render_image' exist in the result.
    print("Testing case [2/3] started.")
    assert 'detected_cells' in result_1 and 'render_image' in result_1, f"Test case [2/3] failed: 'detected_cells' or 'render_image' keys are missing in the result"

    # Testing case 3: Check if image_path is error-prone.
    print("Testing case [3/3] started.")
    try:
        detect_blood_cells('non_existent.jpg')
    except ValueError as e:
        assert str(e) == 'Unable to load image or incorrect path.', f"Test case [3/3] failed: Unintended error message {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_blood_cells()