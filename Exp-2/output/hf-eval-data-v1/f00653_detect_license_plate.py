import yolov5

# Function to detect license plates in car images

def detect_license_plate(img_path):
    '''
    This function uses a pre-trained YOLOv5 model to detect license plates in car images.
    
    Parameters:
    img_path (str): The path or URL to the car image.
    
    Returns:
    dict: A dictionary containing the bounding boxes, scores, and categories of the detected license plates.
    '''
    # Load the pre-trained model
    model = yolov5.load('keremberke/yolov5m-license-plate')
    
    # Configure model parameters
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    
    # Process the input image
    results = model(img_path, size=640)
    
    # Extract the predictions
    predictions = results.pred[0]
    
    # Extract the bounding boxes, scores, and categories
    boxes = predictions[:, :4].tolist()
    scores = predictions[:, 4].tolist()
    categories = predictions[:, 5].tolist()
    
    # Return the results
    return {'boxes': boxes, 'scores': scores, 'categories': categories}