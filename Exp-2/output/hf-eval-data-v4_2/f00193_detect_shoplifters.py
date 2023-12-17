# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_shoplifters(image_path):
    """
    Detects potential shoplifters in the given image by using a pre-trained YOLOv5 object detection model.

    Args:
        image_path (str): The path to the surveillance camera image.

    Returns:
        list: A list of detected objects with bounding boxes, confidence scores, and categories.

    Raises:
        FileNotFoundError: If the given image_path does not exist.
        RuntimeError: If there is an error during the detection process.
    """
    # Load the pre-trained YOLOv5 model
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    # Configure the model parameters
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    # Perform object detection
    results = model(image_path)
    predictions = results.pred[0]
    boxes = predictions[:, :4].tolist()
    scores = predictions[:, 4].tolist()
    categories = predictions[:, 5].tolist()
    # Combine the results
    detected_objects = [{'box': box, 'score': score, 'category': category} for box, score, category in zip(boxes, scores, categories)]
    return detected_objects


# test_function_code --------------------

def test_detect_shoplifters():
    print("Testing started.")
    # Sample surveillance image path
    image_path = "path_to_sample_image.jpg"  # Replace with actual image path
    # Test case 1: Check if the function returns a list
    print("Testing case [1/1] started.")
    detected_objects = detect_shoplifters(image_path)
    assert isinstance(detected_objects, list), f"Test case [1/1] failed: Function should return a list, got {type(detected_objects)}"
    print("Testing finished.")


# call_test_function_line --------------------

test_detect_shoplifters()