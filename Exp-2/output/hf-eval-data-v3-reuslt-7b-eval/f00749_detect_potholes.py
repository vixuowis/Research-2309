# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_potholes(image_path: str) -> dict:
    '''
    Detects potholes in the given image using a pre-trained YOLOv8 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the bounding boxes and masks of the detected potholes.
    '''
    
    # Load image
    img = YOLO(image_path, 1)
        
    # Detect objects in a single image
    results = YOLO(img)
    
    result = render_result(img, results, return_img=True)[0]
    
    return {
        'result': result,
        'boundingboxes': results.xyxyn[0],
        'masks': results.masks[0][None]}

# test_function_code --------------------

def test_detect_potholes():
    '''
    Tests the detect_potholes function with a sample image.
    '''
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_potholes(image_path)
    assert isinstance(result, dict), 'Result should be a dictionary.'
    assert 'boxes' in result, 'Result should contain bounding boxes.'
    assert 'masks' in result, 'Result should contain masks.'
    assert 'render' in result, 'Result should contain a render of the detection.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_potholes()