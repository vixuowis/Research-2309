# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path):
    """
    Detects blood cells in a microscopic image of a blood sample using a YOLOv8 model.
    
    Parameters:
        image_path (str): The file path of the blood sample image.
    
    Returns:
        render: A visualization of the detection results.
    """
    # Load the pre-trained YOLOv8 blood cell detection model
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    # Set model configurations
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Perform object detection
    results = model.predict(image_path)
    # Render and return the visualization
    return render_result(model=model, image=image_path, result=results[0])

# test_function_code --------------------

def test_detect_blood_cells():
    print("Testing blood cell detection.")
    sample_image = 'test_blood_sample.jpg'  # Replace with a valid image path in the test environment

    # Test that the detection function does not raise exceptions and returns a result
    print("Testing detection function.")
    try:
        render = detect_blood_cells(sample_image)
        print("Detection successful.")
    except Exception as e:
        print(f"Detection failed with exception: {e}")
        assert False, "Blood cell detection raised an exception."

    # No robust test for correctness of detection is implemented here due to the lack of ground truth
    # In a real-world scenario, manual or automated validation using labeled data would be necessary
    print("Test completed without exceptions.")

# Run the test
test_detect_blood_cells()