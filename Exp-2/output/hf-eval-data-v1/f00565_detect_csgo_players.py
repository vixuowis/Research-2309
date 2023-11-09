from ultralyticsplus import YOLO, render_result


def detect_csgo_players(image):
    """
    This function detects Counter-Strike: Global Offensive players in the given image using the YOLO object detection model.
    
    Parameters:
    image (str): URL or local path to the image
    
    Returns:
    render: Rendered image with detected players and their bounding boxes
    """
    # Create a YOLO object detection model using the 'keremberke/yolov8n-csgo-player-detection' model
    model = YOLO('keremberke/yolov8n-csgo-player-detection')
    
    # Set the model's parameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    
    # Use the 'predict' method of the YOLO model to obtain the detected players and their bounding boxes
    results = model.predict(image)
    
    # Use the 'render_result' function to visualize the detections on the original image
    render = render_result(model=model, image=image, result=results[0])
    
    return render