from ultralyticsplus import YOLO, render_result
from PIL import Image

# Function to detect potholes in an image using YOLOv8 model
# @param image_path: Path to the image file
# @return: Bounding boxes and masks of detected potholes

def detect_potholes(image_path):
    # Load the pre-trained model
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    # Set the required model configurations
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Open the image file
    image = Image.open(image_path)
    # Run the model prediction
    results = model.predict(image)
    # Return the bounding boxes and masks of detected potholes
    return results[0].boxes, results[0].masks