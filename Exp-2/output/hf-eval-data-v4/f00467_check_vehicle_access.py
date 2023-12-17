# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def check_vehicle_access(image_path, authorized_vehicles):
    # Load the model for license plate detection
    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    # Process the image
    results = model(image_path, size=640)
    predictions = results.pred[0]
    license_plates = []

    # Extract the bounding boxes, scores, and categories
    for *xyxy, conf, cls in predictions:
        # Placeholder for extracting the actual license plate number of each detection
        # Assuming the function extract_license_plate is available, which returns the license plate text
        license_plate_number = extract_license_plate(image_path, xyxy)
        license_plates.append(license_plate_number)

    # Check for authorized vehicles
    for license_plate in license_plates:
        if license_plate in authorized_vehicles:
            return ('Access granted', license_plate)

    # If no authorized vehicles are detected
    return ('Access denied', None)

# test_function_code --------------------

def test_check_vehicle_access():
    print('Testing check_vehicle_access function.')

    # Test data: path to a sample image and list of authorized vehicles
    sample_image_path = 'tests/sample_parking_lot_image.jpg'
    authorized_vehicles_list = ['XYZ123', 'ABC789', 'LMN456']

    # Test case: vehicle is authorized
    print('Testing case [Authorized vehicle].')
    result, plate = check_vehicle_access(sample_image_path, authorized_vehicles_list)
    assert result == 'Access granted' and plate in authorized_vehicles_list, f'Test case [Authorized vehicle] failed: {plate} should be authorized.'

    # Test case: vehicle is not authorized
    print('Testing case [Unauthorized vehicle].')
    result, plate = check_vehicle_access(sample_image_path, [])
    assert result == 'Access denied' and plate is None, 'Test case [Unauthorized vehicle] failed: Access should be denied.'

    print('All tests passed.')

# Run the test function
test_check_vehicle_access()