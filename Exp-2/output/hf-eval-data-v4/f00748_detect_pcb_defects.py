# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_pcb_defects(image_path):
    """
    Detects defects in a given PCB image using a pre-trained YOLO v8 model and returns the annotated image.

    Parameters:
    image_path (str): URL or local path of the PCB image to analyze.

    Returns:
    An object containing the annotated image with detected defects.
    """
    # Initialize the YOLO model for defect detection
    model = YOLO('keremberke/yolov8m-pcb-defect-segmentation')
    
    # Set default parameters for the model
    model.overrides['conf'] = 0.25  # confidence threshold
    model.overrides['iou'] = 0.45   # intersection over union threshold
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000  # maximum number of detections
    
    # Run prediction on the provided image_path
    results = model.predict(image_path)

    # Process the results to obtain a rendered image with defects marked
    render = render_result(model=model, image=image_path, result=results[0])
    return render

# test_function_code --------------------

def test_detect_pcb_defects():
    print("Testing started.")
    # Mock data representing a dataset
    dataset = ["https://example.com/path/to/pcb_image.jpg"]
    sample_data = dataset[0]  # Example image URL

    # Test case 1: Check if function runs without errors
    print("Testing case [1/1] started.")
    try:
        annotated_image = detect_pcb_defects(sample_data)
        assert annotated_image is not None, "Test case [1/1] failed: Function did not return an annotated image."
        print("Test case [1/1] passed.")        
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")

    print("Testing finished.")

# Run the test function
test_detect_pcb_defects()