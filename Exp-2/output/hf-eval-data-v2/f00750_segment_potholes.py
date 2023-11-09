# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def segment_potholes(image):
    """
    This function uses a YOLOv8 model to segment potholes in an image.

    Args:
        image (str): URL or local path of the image containing potholes.

    Returns:
        A tuple (boxes, masks, render), where:
            boxes (list): List of bounding boxes for detected potholes.
            masks (list): List of segmentation masks for detected potholes.
            render (PIL.Image): Image with segmented potholes visualized.
    """
    model = YOLO('keremberke/yolov8m-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image)
    render = render_result(model=model, image=image, result=results[0])
    return results[0].boxes, results[0].masks, render

# test_function_code --------------------

def test_segment_potholes():
    """
    This function tests the segment_potholes function with a sample image.
    """
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    boxes, masks, render = segment_potholes(image)
    assert len(boxes) > 0, 'No potholes detected.'
    assert len(masks) > 0, 'No segmentation masks generated.'
    assert render is not None, 'No rendered image generated.'

# call_test_function_code --------------------

test_segment_potholes()