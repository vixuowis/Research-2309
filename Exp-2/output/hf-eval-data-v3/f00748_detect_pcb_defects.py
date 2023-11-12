# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_pcb_defects(image_path):
    """
    Detect defects of PCB boards from an image using YOLO model.

    Args:
        image_path (str): URL or local path to the image.

    Returns:
        None. The function will display the processed image with the detected defects marked.

    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    """
    model = YOLO('keremberke/yolov8m-pcb-defect-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()

# test_function_code --------------------

def test_detect_pcb_defects():
    """
    Test the detect_pcb_defects function.
    """
    # Test with a URL image
    detect_pcb_defects('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg')
    # Test with a local image
    # detect_pcb_defects('/path/to/local/image.jpg')
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_pcb_defects()