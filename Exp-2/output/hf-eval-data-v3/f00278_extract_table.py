# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def extract_table(image_url: str, model_name: str = 'keremberke/yolov8n-table-extraction', conf: float = 0.25, iou: float = 0.45, agnostic_nms: bool = False, max_det: int = 1000):
    """
    Extracts a table from a given image using the YOLO model.

    Args:
        image_url (str): The URL of the image from which to extract the table.
        model_name (str, optional): The name of the YOLO model to use. Defaults to 'keremberke/yolov8n-table-extraction'.
        conf (float, optional): The confidence threshold for the YOLO model. Defaults to 0.25.
        iou (float, optional): The IoU threshold for the YOLO model. Defaults to 0.45.
        agnostic_nms (bool, optional): Whether to use agnostic non-maximum suppression. Defaults to False.
        max_det (int, optional): The maximum number of detections for the YOLO model. Defaults to 1000.

    Returns:
        A rendered image with the extracted table.
    """
    model = YOLO(model_name)
    model.overrides['conf'] = conf
    model.overrides['iou'] = iou
    model.overrides['agnostic_nms'] = agnostic_nms
    model.overrides['max_det'] = max_det
    results = model.predict(image_url)
    render = render_result(model=model, image=image_url, result=results[0])
    return render

# test_function_code --------------------

def test_extract_table():
    """
    Tests the extract_table function.
    """
    image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = extract_table(image_url)
    assert isinstance(render, type(None)), 'Test Case 1 Failed'
    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_table()