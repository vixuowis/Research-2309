from ultralyticsplus import YOLO, render_result


def detect_blood_cells(image_path):
    """
    This function detects blood cells in an image using the YOLO object detection model.
    The model is trained to detect platelets, red blood cells, and white blood cells.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    render: The image with detected blood cells highlighted.
    """
    # Load the YOLO model trained for blood cell detection
    model = YOLO('keremberke/yolov8n-blood-cell-detection')
    
    # Set the model overrides
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    
    # Use the model to predict the blood cells in the image
    results = model.predict(image_path)
    
    # Render the results on the image
    render = render_result(model=model, image=image_path, result=results[0])
    
    return render