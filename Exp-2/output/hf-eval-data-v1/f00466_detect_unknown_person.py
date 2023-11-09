from ultralyticsplus import YOLO

# Function to detect unknown person in the surveillance image
# This function uses YOLO object detection model from Hugging Face Transformers
# The model is trained on 'keremberke/yolov8m-valorant-detection' dataset
# It detects dropped spike, enemy, planted spike, and teammate objects
# The function takes an image as input and returns the detected objects

def detect_unknown_person(surveillance_image):
    # Instantiate the YOLO class with the model
    model = YOLO('keremberke/yolov8m-valorant-detection')
    # Configure the model's hyperparameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Predict the objects present in the image
    results = model.predict(surveillance_image)
    # Return the detected objects
    return results[0].boxes