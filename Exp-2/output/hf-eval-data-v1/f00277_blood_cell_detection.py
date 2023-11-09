from ultralyticsplus import YOLO, render_result

# Function to detect and count blood cells in a digital blood sample
# Uses the YOLO object detection model from the Hugging Face Transformers library
# The model is trained to detect platelets, red blood cells, and white blood cells

def blood_cell_detection(image):
    # Create a YOLO object detection model for blood cell detection
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    # Set the model's parameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    # Use the model to predict the presence and location of blood cells in the image
    results = model.predict(image)
    # Print the results (bounding boxes and class names)
    print(results[0].boxes)
    # Use the render_result function to visualize the detected objects within the image
    render = render_result(model=model, image=image, result=results[0])
    render.show()
    return results