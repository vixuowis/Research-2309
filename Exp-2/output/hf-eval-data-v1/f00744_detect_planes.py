from ultralyticsplus import YOLO, render_result

# Function to detect planes in an image using the YOLO model
# @param image_path: The path to the image file
# @return: The detection results

def detect_planes(image_path):
    # Create an instance of the YOLO class with the 'keremberke/yolov8m-plane-detection' model
    model = YOLO('keremberke/yolov8m-plane-detection')
    # Set the configuration parameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Load the image data
    image = image_path
    # Use the YOLO model to predict the presence and location of airplanes in the image
    results = model.predict(image)
    # Visualize the results
    rendered = render_result(model=model, image=image, result=results[0])
    rendered.show()
    return results