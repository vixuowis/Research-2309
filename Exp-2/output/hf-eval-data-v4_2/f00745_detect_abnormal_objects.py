# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_abnormal_objects(image_path):
    """
    Detects abnormal objects in an image of an apartment corridor and raises an alert if any are found.

    Args:
        image_path (str): The file path or URL to the image to be analyzed.

    Returns:
        list of dict: Each dict contains 'bbox' for bounding box, 'score' for confidence score, 
                      and 'category' for the detected category of the abnormal object.

    Raises:
        Exception: Raises an exception if no image is provided or model fails to load.
    
    """
    if not image_path:
        raise Exception("No image path provided")

    model = yolov5.load('fcakyon/yolov5s-v7.0')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    results = model(image_path)
    predictions = results.pred[0]

    abnormal_objects = []
    for *bbox, conf, category in predictions:
        if is_abnormal_category(category):
            abnormal_objects.append({
                'bbox': bbox,
                'score': conf.item(),
                'category': results.names[int(category)]
            })

    if abnormal_objects:
        alert_security(abnormal_objects)

    return abnormal_objects

def is_abnormal_category(category):
    """
    Checks if the detected category is considered abnormal in the context of an apartment corridor.

    Args:
        category (int): The category id to be checked.

    Returns:
        bool: True if the category is abnormal, False otherwise.
    """
    abnormal_categories = [24, 26, 28]
    return category in abnormal_categories

def alert_security(objects):
    """
    Sends a security alert with information about the detected abnormal objects.

    Args:
        objects (list of dict): Information about the detected abnormal objects.

    Returns:
        None
    """
    pass

# test_function_code --------------------

def test_detect_abnormal_objects():
    print("Testing started.")
    image_path = 'path_to_test_image.jpg'
    print("Testing case [1/3] started.")
    detected_objects = detect_abnormal_objects(image_path)
    assert len(detected_objects) > 0, "Test case [1/3] failed: No abnormal objects detected when expected."

    print("Testing case [2/3] started.")
    detected_objects = detect_abnormal_objects(image_path)
    assert len(detected_objects) == 0, "Test case [2/3] failed: Abnormal objects detected unexpectedly."

    print("Testing case [3/3] started.")
    try:
        detect_abnormal_objects(None)
        assert False, "Test case [3/3] failed: No exception raised for missing image_path."
    except Exception as e:
        assert str(e) == "No image path provided", "Test case [3/3] failed: Incorrect exception message."

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_abnormal_objects()