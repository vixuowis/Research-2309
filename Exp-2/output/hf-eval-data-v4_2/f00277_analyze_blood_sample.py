# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def analyze_blood_sample(image_path_or_url):
    """
    Analyzes a blood sample image to detect and count platelets, red blood cells, and white blood cells.

    Args:
        image_path_or_url (str): The file path or URL to the image of the blood sample.

    Returns:
        dict: A dictionary with counts of 'platelets', 'red_blood_cells', and 'white_blood_cells'.

    Raises:
        ValueError: If the image_path_or_url is not provided or is invalid.
    """
    if not image_path_or_url:
        raise ValueError('The image path or URL must be provided')

    # Initialize the model with the specified weight
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Perform object detection
    results = model.predict(image_path_or_url)

    # Analyzing the results
    platelets_count = sum(1 for i in results[0].classes if i == 'platelets')
    red_blood_cells_count = sum(1 for i in results[0].classes if i == 'red blood cells')
    white_blood_cells_count = sum(1 for i in results[0].classes if i == 'white blood cells')

    # Counting and returning results
    return {
        'platelets': platelets_count,
        'red_blood_cells': red_blood_cells_count,
        'white_blood_cells': white_blood_cells_count
    }

# test_function_code --------------------

def test_analyze_blood_sample():
    print("Testing started.")
    image_path_or_url = 'path/to/sample/blood_image.jpg'  # Substitute with an actual image path or URL

    # Testing case 1: Analyze valid image
    print("Testing case [1/1] started.")
    results = analyze_blood_sample(image_path_or_url)
    assert isinstance(results, dict), f"Test case [1/1] failed: Results should be a dictionary, got {type(results)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_blood_sample()