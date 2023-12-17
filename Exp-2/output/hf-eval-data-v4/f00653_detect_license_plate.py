# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def detect_license_plate(image_path, conf_threshold=0.25, iou_threshold=0.45, size=640):
    """
    Detect license plates in a car image using a pretrained YOLOv5 model.
    
    Parameters:
    - image_path: str, the path or URL to the car image
    - conf_threshold: float, confidence threshold for the detection
    - iou_threshold: float, IoU threshold for the detection
    - size: int, the size to which to resize the images for processing
    
    Returns:
    - A list containing tuples of detected license plate bounding boxes and their scores
    """
    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = conf_threshold  # Set confidence threshold
    model.iou = iou_threshold  # Set IoU threshold
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    
    results = model(image_path, size=size)
    predictions = results.pred[0]
    plates_info = [(tuple(box), score) for box, score in zip(predictions[:, :4], predictions[:, 4]) if score > conf_threshold]
    
    # Display and save the results if needed (comment out or remove if not required)
    # results.show()
    # results.save(save_dir='results/')
    
    return plates_info


# test_function_code --------------------

def test_detect_license_plate():
    print("Testing started.")
    # As we don't have a real dataset, replace the following line with an actual image path or URL
    sample_image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    
    # Testing case 1: Basic functionality with default parameters
    print("Testing case [1/3] started.")
    results = detect_license_plate(sample_image_path)
    assert len(results) > 0, f"Test case [1/3] failed: No license plates detected."
    
    # Testing case 2: Custom confidence threshold
    print("Testing case [2/3] started.")
    custom_conf_threshold = 0.5
    results_high_conf = detect_license_plate(sample_image_path, conf_threshold=custom_conf_threshold)
    assert all(score > custom_conf_threshold for _, score in results_high_conf), f"Test case [2/3] failed: Detected plates with confidence below the threshold."
    
    # Testing case 3: Detecting with a different image size
    print("Testing case [3/3] started.")
    different_size = 320
    results_diff_size = detect_license_plate(sample_image_path, size=different_size)
    assert len(results_diff_size) > 0, f"Test case [3/3] failed: No license plates detected with size {different_size}."
    
    print("Testing finished.")

# Run the test function
test_detect_license_plate()
