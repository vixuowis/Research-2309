# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_unauthorized_objects(image_path):
    """
    Detect unauthorized or suspicious objects in an image.

    Parameters:
        image_path (str): Path to the image to be analyzed.

    Returns:
        bool: True if unauthorized objects are detected, otherwise False.
    """
    # Define the list of unauthorized or suspicious objects
    unauthorized_objects = ['person', 'knife', 'gun']
    # Set model parameters
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # IOU threshold for NMS
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    # Load image
    img = image_path
    # Run detection
    results = model(img, size=640, augment=True)
    # Extract predictions
    predictions = results.pred[0]
    # Check for unauthorized objects and report
    for *box, conf, cls in predictions:
        category = model.names[int(cls)]
        if category in unauthorized_objects and conf >= model.conf:
            print(f"Unauthorized object detected: {category}")
            return True
    return False

# test_function_code --------------------

def test_detect_unauthorized_objects():
    print("Testing started.")
    image_path_authorized = 'path_to_image_with_no_unauthorized_objects.jpg'
    image_path_unauthorized = 'path_to_image_with_unauthorized_objects.jpg'

    print("Testing case [1/2] started.")
    assert not detect_unauthorized_objects(image_path_authorized), "Test case [1/2] failed: Unauthorized object detected in authorized image."

    print("Testing case [2/2] started.")
    assert detect_unauthorized_objects(image_path_unauthorized), "Test case [2/2] failed: Unauthorized object not detected when expected."

    print("Testing finished.")

test_detect_unauthorized_objects()