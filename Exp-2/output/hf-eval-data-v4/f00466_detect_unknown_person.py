# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO

# function_code --------------------

def detect_unknown_person(image_path, known_persons):
    '''
    Detect if any unknown person is present in the image.

    Parameters:
        image_path (str): The path to the surveillance image.
        known_persons (list): A list of known individuals to compare against.

    Returns:
        bool: True if an unknown person is detected, False otherwise.
    '''
    # Load the YOLO model
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Predict objects in the image
    results = model.predict(image_path)

    # Check for unknown persons
    for obj in results[0].xyxy:
        if obj[-1] == 'person' and obj not in known_persons:
            return True
    return False

# test_function_code --------------------

def test_detect_unknown_person():
    print("Testing detect_unknown_person function.")
    known_persons = ['person1', 'person2']
    image_path = 'path_to_test_image_with_unknown_person.jpg'

    # Test case: Unknown person detected
    print("Testing case: Unknown person detected.")
    assert detect_unknown_person(image_path, known_persons) == True, "Test case failed: Unknown person not detected as expected."

    image_path = 'path_to_test_image_with_no_unknown_person.jpg'

    # Test case: No unknown person detected
    print("Testing case: No unknown person detected.")
    assert detect_unknown_person(image_path, known_persons) == False, "Test case failed: False positive detected."
    print("All tests passed.")

# Run the test function
test_detect_unknown_person()