from ultralyticsplus import YOLO, render_result


def detect_potholes(image_path):
    """
    This function uses a pre-trained YOLOv8 model to detect potholes in an image.
    The model is trained on the 'pothole-segmentation' dataset and is capable of
    detecting potholes with high accuracy.

    Args:
        image_path (str): The path to the image file.

    Returns:
        render: The original image with the detected potholes highlighted.
    """
    # Load the pre-trained YOLOv8 model
    model = YOLO('keremberke/yolov8s-pothole-segmentation')

    # Set the desired model configuration parameters
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Pass the image through the model
    results = model.predict(image_path)

    # Render the segmentation results with the original image
    render = render_result(model=model, image=image_path, result=results[0])

    return render