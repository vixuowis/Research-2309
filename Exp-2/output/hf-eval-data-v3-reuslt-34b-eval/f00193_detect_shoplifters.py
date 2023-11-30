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

    # Detect shoplifters in image
    results = yolov5.model('./yolov5/weights/best.pt').infer(source=image_path)
    
    # Filter out unrelated predictions using the COCO dataset class names (see: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)
    results = [r for r in results.pred[0] if r[5:] not in ['clock', 'scissors', 'keyboard', 'bottle']] # TODO: Filter out more unrelated categories

    return {
        "boxes": [list(i[:4]) for i in results],
        "scores": [float(j) for j in results[:, 4]],
        "classes": [int(k) for k in results[:, 5]]
    }

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