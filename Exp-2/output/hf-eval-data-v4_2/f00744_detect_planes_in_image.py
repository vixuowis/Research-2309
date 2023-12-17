# requirements_file --------------------

pip install ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_planes_in_image(image_path):
    """
    Detects planes in an image using the keremberke/yolov8m-plane-detection model.

    Args:
        image_path (str): The path or URL to the image for plane detection.

    Returns:
        tuple: A tuple containing the detection boxes and a rendered image with detections.

    Raises:
        ValueError: If the image_path is invalid or empty.
    """
    if not image_path:
        raise ValueError('The image_path cannot be empty.')

    # Initialize the YOLO model with the specific plane detection configuration.
    model = YOLO('keremberke/yolov8m-plane-detection')
    model.overrides['conf'] = 0.25  # Confidence threshold
    model.overrides['iou'] = 0.45   # Intersection Over Union threshold
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000  # Maximum detections

    # Predict the presence and location of airplanes in the image.
    results = model.predict(image_path)
    boxes = results[0].boxes

    # Visualize the results.
    rendered_image = render_result(model=model, image=image_path, result=results[0])

    return boxes, rendered_image

# test_function_code --------------------

def test_detect_planes_in_image():
    print("Testing started.")
    sample_image = 'path_to_sample_image.jpg'  # Replace with the path to your sample image

    # Testing case 1: Valid image path
    print("Testing case [1/1] started.")
    boxes, rendered_image = detect_planes_in_image(sample_image)
    assert boxes is not None and rendered_image is not None, f"Test case [1/1] failed: Expected detections and a rendered image, got {boxes}, {rendered_image}."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_planes_in_image()