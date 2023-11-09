# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detect and count platelets, red blood cells, and white blood cells in a digital blood sample image.

    Args:
        image_path (str): The path or URL of the image to analyze.

    Returns:
        dict: A dictionary with the counts of each blood cell type and a rendered image with the detected objects.
    """
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    cell_counts = {
        'platelets': 0,
        'red_blood_cells': 0,
        'white_blood_cells': 0
    }
    for result in results:
        if result.label == 'platelet':
            cell_counts['platelets'] += 1
        elif result.label == 'red_blood_cell':
            cell_counts['red_blood_cells'] += 1
        elif result.label == 'white_blood_cell':
            cell_counts['white_blood_cells'] += 1

    render = render_result(model=model, image=image_path, result=results[0])

    return {'cell_counts': cell_counts, 'render': render}

# test_function_code --------------------

def test_detect_blood_cells():
    """
    Test the detect_blood_cells function with a sample image.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_blood_cells(image_path)
    assert isinstance(result, dict)
    assert 'cell_counts' in result
    assert 'render' in result
    assert isinstance(result['cell_counts'], dict)
    assert isinstance(result['render'], type(render_result))

# call_test_function_code --------------------

test_detect_blood_cells()