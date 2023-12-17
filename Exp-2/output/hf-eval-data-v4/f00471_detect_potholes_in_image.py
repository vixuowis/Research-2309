# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO
from PIL import Image

# function_code --------------------

def detect_potholes_in_image(image_path):
    """Detects potholes in an image using the YOLOv8 model.

    Args:
        image_path (str): The path or URL to the image.

    Returns:
        tuple: A tuple containing the detected potholes' bounding boxes and masks.
    """
    # Load pre-trained YOLOv8 model for pothole detection
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    # Set required model configurations
    model.overrides['conf'] = 0.25  # Confidence threshold
    model.overrides['iou'] = 0.45   # Intersection over Union threshold
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000  # Maximum number of detections
    # Load image
    image = Image.open(image_path)
    # Predict potholes in the image
    results = model.predict(image)
    # Return bounding boxes and masks
    return results[0].boxes, results[0].masks

# test_function_code --------------------

def test_detect_potholes_in_image():
    print("Testing started.")
    # We don't have a real dataset to load, so this is just a placeholder
    # Ideally, a sample image is downloaded or accessed from a known data source
    sample_image_path = 'sample_image.jpg'  # Placeholder path
    
    # Run the detection function on the sample image
    print("Testing detection on sample image.")
    boxes, masks = detect_potholes_in_image(sample_image_path)
    
    # Check if the results are in expected format (list/tuple)
    assert isinstance(boxes, (list, tuple)), "The 'boxes' should be a list or a tuple."
    assert isinstance(masks, (list, tuple)), "The 'masks' should be a list or a tuple."
    
    # Check if the function returns any results, assuming potholes are present in the sample image
    assert len(boxes) > 0, "No potholes detected, result should not be empty."
    assert len(masks) > 0, "No potholes detected, result should not be empty."
    print("Testing finished.")

# Run the test
test_detect_potholes_in_image()