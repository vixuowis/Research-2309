# requirements_file --------------------

!pip install -U ultralyticsplus ultralytics

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_potholes_in_image(image_path):
    """
    Detect potholes in a given image using the YOLOv8 image segmentation model.

    Parameters:
        image_path (str): A URL or a local path to the image to analyze.

    Returns:
        A visualization of the image with detected potholes.
    """
    # Load the pre-trained YOLOv8 image segmentation model
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25  # Set confidence threshold
    model.overrides['iou'] = 0.45   # Set Intersection Over Union threshold
    model.overrides['agnostic_nms'] = False  # Disable class-agnostic NMS
    model.overrides['max_det'] = 1000  # Maximum number of detections

    # Predict potholes in the image
    results = model.predict(image_path)
    
    # Render the segmentation results
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()

    return render

# test_function_code --------------------

def test_detect_potholes_in_image():
    print("Testing detect_potholes_in_image function.")
    sample_image_path = 'path_to_a_test_image.jpg'

    # Execute the function
    result_render = detect_potholes_in_image(sample_image_path)

    # Test case 1: Check if the result render is not None
    print("Testing case [1/1] started.")
    assert result_render is not None, f"Test case [1/1] failed: Function did not return any output"
    print("Testing finished.")

# Run the test function
test_detect_potholes_in_image()