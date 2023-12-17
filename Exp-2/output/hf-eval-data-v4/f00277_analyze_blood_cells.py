# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.24 ultralytics==8.0.23

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def analyze_blood_cells(image_path_or_url):
    # Create a YOLO object detection model
    model = YOLO('keremberke/yolov8m-blood-cell-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    # Predict the presence and location of blood cells
    results = model.predict(image_path_or_url)

    # Extract identified class names
    class_names = results[0].names

    # Count the number of each type of blood cell
    platelets_count = sum([1 for class_name in results[0].names if class_name == 'platelets'])
    red_cells_count = sum([1 for class_name in results[0].names if class_name == 'red blood cell'])
    white_cells_count = sum([1 for class_name in results[0].names if class_name == 'white blood cell'])

    # Return the counts
    return {
        'platelets_count': platelets_count,
        'red_cells_count': red_cells_count,
        'white_cells_count': white_cells_count
    }

# test_function_code --------------------

def test_analyze_blood_cells():
    print('Testing analyze_blood_cells function started.')
    sample_image_url = 'blood_sample_image_path_or_url'  # Replace with an actual image URL for testing

    # Perform the analysis
    result = analyze_blood_cells(sample_image_url)

    # Test case: Check if the result is a dictionary with the required keys
    assert isinstance(result, dict), 'Result should be a dictionary.'
    assert all(key in result for key in ['platelets_count', 'red_cells_count', 'white_cells_count']), 'Result dictionary missing required keys.'

    # Test case: Check if the counts are integers
    assert all(isinstance(count, int) for count in result.values()), 'All counts should be integers.'

    # Test case: Mocked expected result for a sample image (to be replaced with actual expected counts for a real image)
    expected_result = {'platelets_count': 5, 'red_cells_count': 500, 'white_cells_count': 10} # Mocked data
    assert result == expected_result, 'The analyzed counts do not match the expected result.'

    print('Testing analyze_blood_cells function finished successfully.')

# Run the test
if __name__ == '__main__':
    test_analyze_blood_cells()