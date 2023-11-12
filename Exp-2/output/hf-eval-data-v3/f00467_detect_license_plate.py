# function_import --------------------

import yolov5

# function_code --------------------

def detect_license_plate(img_path, authorized_vehicles):
    """
    Detects the license plate from the image and checks if it's an authorized vehicle.

    Args:
        img_path (str): Path to the image.
        authorized_vehicles (list): List of authorized license plates.

    Returns:
        str: 'Access granted' if the vehicle is authorized, 'Access denied' otherwise.
    """
    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    results = model(img_path, size=640)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    vehicle_license_plate = '...'  # Extract license plate number from image

    if vehicle_license_plate in authorized_vehicles:
        return 'Access granted'
    else:
        return 'Access denied'

# test_function_code --------------------

def test_detect_license_plate():
    """
    Tests the detect_license_plate function.
    """
    authorized_vehicles = ['XYZ123', 'ABC789', 'LMN456']
    assert detect_license_plate('path/to/authorized_vehicle.jpg', authorized_vehicles) == 'Access granted'
    assert detect_license_plate('path/to/unauthorized_vehicle.jpg', authorized_vehicles) == 'Access denied'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_license_plate()