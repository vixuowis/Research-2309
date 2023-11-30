# function_import --------------------

import yolov5

# function_code --------------------

def detect_vehicles(image_path):
    """
    Detect vehicles in the given image using YOLOv5 object detection model.

    Args:
        image_path (str): The path or URL to the image.

    Returns:
        dict: A dictionary containing the bounding boxes, scores, and categories of the detected vehicles.
    """
    
    # Instantiate YOLOv5 object detector using model pretrained on COCO dataset (80 classes)
    yolo = yolov5.load('yolov5s')
    
    # Detect objects in an input image with bounding box overlays and verbose mode
    result_dict = yolo(image_path, size=640, show=False, save_txt=False, 
                       line_thickness=2, overlap=0.5)     # 640 is the input image dimension used for training YOLOv5s model (pretrained on COCO dataset)
    
    # Filter the results to return only vehicles with confidence > 80%
    vehicle_boxes = []   # initialize list of detected vehicle bounding boxes
    for i, pred in enumerate(result_dict.pred[0]):
        if (pred is None or pred.names == 'bus' or pred.names == 'truck'):
            continue    # skip any detections that are empty/None or are not a bus or truck
    
        if ((pred.confidence * 100) > 80):   # multiply confidence score by 100 to get percentage, then check if it is greater than 80%
            vehicle_boxes += [i]
        
    vehicles = {
        'bounding_boxes': result_dict.rect[vehicle_boxes],   # bounding boxes of detected vehicles only
        'scores': result_dict.score[vehicle_boxes],          # confidence scores (probabilities) of detection
        'categories': [{'id': 3, 'name': 'vehicle'}] * len(result_dict.rect[vehicle_boxes])   # category name and id for all vehicles detected in the image
    }
        
    return vehicles

# test_function_code --------------------

def test_detect_vehicles():
    """
    Test the detect_vehicles function.
    """
    image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_vehicles(image_url)
    assert isinstance(result, dict)
    assert 'boxes' in result
    assert 'scores' in result
    assert 'categories' in result
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_vehicles()