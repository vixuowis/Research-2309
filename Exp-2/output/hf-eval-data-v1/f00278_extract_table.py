from ultralyticsplus import YOLO, render_result


def extract_table(image):
    '''
    This function extracts a table from a given image using the YOLO model from Hugging Face Transformers.
    
    Parameters:
    image (str): The URL of the image from which the table needs to be extracted.
    
    Returns:
    render: The extracted table.
    '''
    # Instantiate the YOLO model using the 'keremberke/yolov8n-table-extraction' as the model name.
    model = YOLO('keremberke/yolov8n-table-extraction')
    
    # Set the confidence threshold, IoU threshold, agnostic non-maximum suppression, and maximum detections for the YOLO model.
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    
    # Use the YOLO model to make predictions on the loaded image.
    results = model.predict(image)
    
    # Display the extracted table using the 'render_result' function.
    render = render_result(model=model, image=image, result=results[0])
    
    return render