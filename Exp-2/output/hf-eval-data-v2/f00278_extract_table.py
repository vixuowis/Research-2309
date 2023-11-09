# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def extract_table(image_url):
    """
    Extracts a table from a given image using the YOLO model.

    Args:
        image_url (str): The URL of the image from which the table is to be extracted.

    Returns:
        A rendered image with the extracted table.
    """
    model = YOLO('keremberke/yolov8n-table-extraction')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_url)
    render = render_result(model=model, image=image_url, result=results[0])
    return render

# test_function_code --------------------

def test_extract_table():
    """
    Tests the 'extract_table' function by using a sample image.
    """
    image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = extract_table(image_url)
    assert result is not None, 'No table was extracted.'

# call_test_function_code --------------------

test_extract_table()