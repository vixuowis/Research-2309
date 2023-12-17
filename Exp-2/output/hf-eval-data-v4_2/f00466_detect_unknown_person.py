# requirements_file --------------------

pip install ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_unknown_person(image, known_persons):
    """
    Detect if any unknown person enters the property by analyzing surveillance camera images.
    
    Args:
        image (str): The URL or path to the surveillance camera image.
        known_persons (list of str): A list of known person IDs.
    
    Returns:
        bool: True if an unknown person is detected, False otherwise.
    
    Raises:
        ValueError: If the input image is not available or cannot be processed.
    """
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    
    results = model.predict(image)
    person_boxes = [box for box in results[0].boxes if box[6] == 'person']

    for box in person_boxes:
        person_id = get_person_id_from_box(box) # Function to get person ID from detection
        if person_id not in known_persons:
            return True
    return False

# test_function_code --------------------

def test_detect_unknown_person():
    print("Testing started.")
    # Assume 'load_dataset' and 'get_person_id_from_box' are pre-defined.
    dataset = load_dataset('surveillance_dataset')
    sample_data = dataset[0]
    known_persons = ['person1', 'person2']

    # Test case 1: Known person detected
    print("Testing case [1/3] started.")
    assert not detect_unknown_person(sample_data, known_persons), "Test case [1/3] failed: Known person detected as unknown."

    # Test case 2: Unknown person detected
    print("Testing case [2/3] started.")
    sample_data_unknown = dataset[1] # An image with an unknown person
    assert detect_unknown_person(sample_data_unknown, known_persons), "Test case [2/3] failed: Unknown person not detected."

    # Test case 3: No person detected
    print("Testing case [3/3] started.")
    sample_data_no_person = dataset[2] # An image without people
    assert not detect_unknown_person(sample_data_no_person, known_persons), "Test case [3/3] failed: False positive detection."
    print("Testing finished.")

# Run the test function
# test_detect_unknown_person()

# call_test_function_line --------------------

test_detect_unknown_person()