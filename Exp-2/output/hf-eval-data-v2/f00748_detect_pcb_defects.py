# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_pcb_defects(image):
    """
    Detects defects of PCB boards from an image using YOLO model.

    Args:
        image (str): URL or local path to the image.

    Returns:
        render: A render object that can be displayed using render.show().
    """
    model = YOLO('keremberke/yolov8m-pcb-defect-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image)
    render = render_result(model=model, image=image, result=results[0])
    return render

# test_function_code --------------------

def test_detect_pcb_defects():
    """
    Tests the detect_pcb_defects function.
    """
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = detect_pcb_defects(image)
    assert isinstance(render, type(render_result())), 'The return type is not correct.'

# call_test_function_code --------------------

test_detect_pcb_defects()