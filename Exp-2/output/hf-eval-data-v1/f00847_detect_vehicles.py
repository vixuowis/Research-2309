import yolov5


def detect_vehicles(image_path):
    """
    Detect vehicles in an image using the YOLOv5 object detection model.

    Args:
        image_path (str): The path or URL to the image.

    Returns:
        A tuple (boxes, scores, categories), where:
            boxes (list): A list of bounding boxes for detected vehicles.
            scores (list): A list of confidence scores for the detected vehicles.
            categories (list): A list of categories for the detected vehicles.
    """
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    results = model(image_path, size=640, augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    return boxes, scores, categories