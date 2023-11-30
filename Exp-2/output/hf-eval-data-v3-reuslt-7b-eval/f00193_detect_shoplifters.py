# function_import --------------------

import yolov5

# function_code --------------------

def detect_shoplifters(image_path: str) -> dict:
    '''
    Detect potential shoplifters in the given image using the pre-trained YOLOv5 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the detected objects' bounding boxes, scores, and categories.
    '''

    # Initialize a model with the pretrained weights in PyTorch Hub
    model = yolov5.load('yolov5s', pretrained=True)

    # Run inference on the image
    results = model(image_path)
    
    # Convert the output into a dictionary containing bounding boxes, scores, and categories for each object in the image
    outputs = results.pandas().xyxy[0].to_dict()
    
    return outputs


# test_function_code --------------------

def test_detect_shoplifters():
    '''
    Test the detect_shoplifters function.
    '''
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = detect_shoplifters(image_path)
    assert isinstance(results, dict)
    assert 'boxes' in results
    assert 'scores' in results
    assert 'categories' in results
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_shoplifters()