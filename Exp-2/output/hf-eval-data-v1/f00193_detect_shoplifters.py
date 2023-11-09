import yolov5

# Function to detect shoplifters using yolov5 object detection model

def detect_shoplifters(image_path):
    """
    This function takes an image path as input and uses the yolov5 object detection model to detect potential shoplifters.
    It returns the bounding boxes, scores, and categories of the detected objects.
    
    Parameters:
    image_path (str): The path to the surveillance image.
    
    Returns:
    dict: A dictionary containing the bounding boxes, scores, and categories of the detected objects.
    """
    # Load the pre-trained model
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    # Set the model's confidence threshold and intersection over union (IoU) threshold
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    # Pass the image to the model and obtain object detection results
    results = model(image_path)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    # Return the results
    return {'boxes': boxes, 'scores': scores, 'categories': categories}