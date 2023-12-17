# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_blood_cells(image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Detect blood cells in an image using YOLOv8 model.

    Parameters:
        image_path (str): The path to the image to be analyzed.
        conf_threshold (float): The confidence threshold for the model to detect objects.
        iou_threshold (float): The IoU threshold for non-maximum suppression.

    Returns:
        tuple: A tuple containing the model's predictions and the rendered image.
    """
    # Load the pretrained model
    model = YOLO('keremberke/yolov8n-blood-cell-detection')
    model.overrides['conf'] = conf_threshold
    model.overrides['iou'] = iou_threshold

    # Predict blood cells in the image
    results = model.predict(image_path)

    # Render the prediction results on the image
    render = render_result(model=model, image=image_path, result=results[0])
    return results[0].boxes, render

# test_function_code --------------------

def test_detect_blood_cells():
    print("Testing started.")
    # Assuming 'some_image.jpg' is an image in the test dataset
    sample_image = 'some_image.jpg'

    # Test case 1: Check if function returns a tuple
    print("Testing case [1/1] started.")
    boxes, render = detect_blood_cells(sample_image)
    assert isinstance(boxes, list) and hasattr(render, 'show'), "Test case [1/1] failed: The function should return a tuple consisting of detected boxes and a rendered image."
    print("Testing finished.")

# Run the test function
test_detect_blood_cells()