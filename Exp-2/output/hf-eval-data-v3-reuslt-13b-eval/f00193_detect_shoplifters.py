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
    
    # Detect potential shoplifters in the given image.
    results = yolov5(image_path)

    # Format the detected objects into a dictionary.
    formatted_results = {
        'boxes': [],
        'scores': [],
        'categories': []
    }

    for result in results['common']['objects']:
        formatted_results['boxes'].append(result["relative_coordinates"])
        formatted_results['scores'].append(result["confidence"])
        formatted_results['categories'].append('shoplifting' if result["class"] == 0 else 'not shoplifting')
    
    return formatted_results

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