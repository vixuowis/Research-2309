# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_objects_in_valorant(image_path):
    # Load the pre-trained YOLOv8 model for Valorant object detection
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25 # Confidence threshold
    model.overrides['iou'] = 0.45 # Intersection over Union threshold
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000 # Max number of detections per image

    # Perform object detection
    results = model.predict(image_path)

    # Render and show the detection results
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()

    # Return the detection results
    return results[0]

# test_function_code --------------------

def test_detect_objects_in_valorant():
    print('Testing detect_objects_in_valorant function...')
    # Download a sample Valorant game frame image for testing
    sample_image_url = 'https://example.com/sample_valorant_frame.jpg'  # Replace with an actual image URL
    image_path = download_image(sample_image_url)  # Replace with actual download function

    # Test object detection on the sample image
    results = detect_objects_in_valorant(image_path)

    # Verify that results are not empty
    assert len(results.boxes) > 0, 'No objects detected.'
    print('Test passed.')

# Run the test function
test_detect_objects_in_valorant()