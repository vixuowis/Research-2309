# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def extract_table_from_image(image_path):
    # Initialize the YOLO model with the table-extraction configuration.
    model = YOLO('keremberke/yolov8n-table-extraction')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Make predictions on the provided image.
    results = model.predict(image_path)

    # Print the boxes of the detected tables and show the annotated image.
    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()

    # Return the results.
    return results[0]

# test_function_code --------------------

def test_extract_table_from_image():
    print("Testing started.")
    # We assume an image URL for testing purposes.
    test_image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

    # Test case 1: Ensure excepted boxes are returned for the test image
    print("Testing case [1/3] started.")
    expected_boxes = [] # Replace with expected boxes for the test image.
    results = extract_table_from_image(test_image_url)
    assert results.boxes == expected_boxes, f"Test case [1/3] failed: Expected boxes do not match with the results"

    # Further test cases can be included here if necessary.

    print("Testing finished.")

# Run the test function
test_extract_table_from_image()