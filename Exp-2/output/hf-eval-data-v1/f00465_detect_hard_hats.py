from ultralyticsplus import YOLO, render_result

# Function to detect hard hats in an image
# @param image_path: Path to the image
# @return: Detection results

def detect_hard_hats(image_path):
    # Create an instance of the YOLO model
    model = YOLO('keremberke/yolov8m-hard-hat-detection')
    # Set the confidence threshold
    model.overrides['conf'] = 0.25
    # Set the Intersection over Union (IoU) threshold
    model.overrides['iou'] = 0.45
    # Set the agnostic_nms to False
    model.overrides['agnostic_nms'] = False
    # Set the maximum detections
    model.overrides['max_det'] = 1000
    # Predict the results
    results = model.predict(image_path)
    # Print the bounding boxes
    print(results[0].boxes)
    # Render the results
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()
    return results