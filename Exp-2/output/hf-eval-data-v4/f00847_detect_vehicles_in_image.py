# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_vehicles_in_image(image_path):
    """
    Detects vehicles in the given image using the pre-trained YOLOv5 model.

    Args:
        image_path (str): The path or URL to the image to be processed.

    Returns:
        list: A list of detected vehicle bounding boxes, scores, and categories.
    """
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    
    results = model(image_path, size=640, augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4].tolist()  # Convert bounding boxes to list
    scores = predictions[:, 4].tolist()  # Convert scores to list
    categories = predictions[:, 5].tolist()  # Convert categories to list
    
    return [{'bbox': box, 'score': score, 'category': category} for box, score, category in zip(boxes, scores, categories)]

# test_function_code --------------------

def test_detect_vehicles_in_image():
    print('Testing detect_vehicles_in_image function.')

    # Assuming 'test_image.jpg' is an image from the dataset
    image_path = 'test_image.jpg'
    detections = detect_vehicles_in_image(image_path)

    # The exact output depends on the image, but we can check types and structure
    assert isinstance(detections, list), 'The result should be a list.'
    for detection in detections:
        assert 'bbox' in detection and isinstance(detection['bbox'], list), 'Each detection should include a bbox list.'
        assert 'score' in detection and isinstance(detection['score'], float), 'Each detection should include a score float.'
        assert 'category' in detection and isinstance(detection['category'], float), 'Each detection should include a category float.'

    print('All tests passed.')