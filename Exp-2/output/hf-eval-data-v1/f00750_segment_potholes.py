from ultralyticsplus import YOLO, render_result


def segment_potholes(image: str) -> dict:
    """
    Function to segment potholes in an image using a pretrained YOLOv8 model.

    Args:
        image (str): URL or local path of the image containing potholes.

    Returns:
        dict: A dictionary containing the bounding boxes and masks of the detected potholes.
    """
    model = YOLO('keremberke/yolov8m-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image)
    return {'boxes': results[0].boxes, 'masks': results[0].masks}