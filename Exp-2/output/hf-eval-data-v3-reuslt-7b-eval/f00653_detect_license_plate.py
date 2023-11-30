# function_import --------------------

import yolov5

# function_code --------------------

def detect_license_plate(img_path: str) -> dict:
    """
    Detects license plates in the given image using a pre-trained YOLOv5 model.

    Args:
        img_path (str): The path or URL to the image.

    Returns:
        dict: A dictionary containing the detected license plates' bounding boxes, scores, and categories.
    """
    
    # Load a pre-trained YOLOv5 model
    # Note that this will only work on GPUs or TPUs since it uses CUDA by default
    yolo = yolov5.load('yolov5x6.pt')
    yolo.conf = 0.27
    yolo.iou = 0.45
    
    # Run inference on the image with a model and return the predictions as an object detection prediction (list of dictionaries)
    preds = yolo([img_path])  # Returns a list[List[Dict]]
    
    # Extract relevant info for license plate detections into a dictionary
    license_plates = [{
        'bbox': x['bbox'],
        'score': x['conf'],
        'category': x['cls']} for x in preds[0] if x['cls'] == 85] # 85 is the class ID of a license plate (see below)
    
    return license_plates


# test_function_code --------------------

def test_detect_license_plate():
    """
    Tests the detect_license_plate function with a few test cases.
    """
    test_img1 = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    test_img2 = 'https://placekitten.com/200/300'

    result1 = detect_license_plate(test_img1)
    result2 = detect_license_plate(test_img2)

    assert isinstance(result1, dict), 'Result should be a dictionary.'
    assert isinstance(result2, dict), 'Result should be a dictionary.'
    assert 'boxes' in result1, 'Result dictionary should contain boxes.'
    assert 'scores' in result1, 'Result dictionary should contain scores.'
    assert 'categories' in result1, 'Result dictionary should contain categories.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_license_plate()