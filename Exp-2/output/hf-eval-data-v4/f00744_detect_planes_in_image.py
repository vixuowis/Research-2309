# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_planes_in_image(image_path):
    """
    Detect airplanes in an image using a pre-trained YOLOv8 model.

    :param image_path: str. The path to the image to analyze.
    :return: A tuple with the detection results and the rendered image.
    """
    # Initialize the model with the plane detection configuration
    model = YOLO('keremberke/yolov8m-plane-detection')

    # Set override parameters for the model
    model.overrides['conf'] = 0.25  # Confidence threshold
    model.overrides['iou'] = 0.45   # Intersection over Union threshold
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000  # Maximum number of detections

    # Perform prediction using the specified image
    results = model.predict(image_path)

    # Get bounding boxes from the results
    boxes = results[0].boxes

    # Render and display the result
    rendered = render_result(model=model, image=image_path, result=results[0])

    # Return the bounding boxes and the rendered image
    return boxes, rendered

# test_function_code --------------------

def test_detect_planes_in_image():
    print("Testing detect_planes_in_image() function.")

    # Note: The following dataset and image should be replaced with actual data in a real implementation
    sample_image_path = 'path_to_sample_image.jpg'  # Replace with a valid image path

    # Test case 1: Check if the function returns a tuple
    print("Test case 1: Checking return type.")
    result, rendered = detect_planes_in_image(sample_image_path)
    assert isinstance(result, list), "The detection result should be in list format."
    assert isinstance(rendered, object), "The rendered image should be an object."
    print("Test case 1 passed.")

    # Test case 2: Check if the function detects at least one plane
    print("Test case 2: Checking for at least one detection.")
    assert len(result) > 0, "The function should detect at least one airplane in the image."
    print("Test case 2 passed.")

    print("Testing for detect_planes_in_image() function completed successfully.")

# Run the test function
test_detect_planes_in_image()