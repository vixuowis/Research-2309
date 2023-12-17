# requirements_file --------------------

!pip install -U yolov5

# function_import --------------------

import yolov5

# function_code --------------------

def authorize_vehicle_access(image_path, authorized_vehicles):
    """
    Authorize vehicle access based on license plate detection in an image.

    Args:
        image_path (str): The path to the image of the vehicle's license plate.
        authorized_vehicles (list): A list of authorized vehicle license plates.

    Returns:
        bool: True if the vehicle is authorized, False otherwise.

    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    model = yolov5.load('keremberke/yolov5m-license-plate')
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

    results = model(image_path, size=640)
    predictions = results.pred[0]

    # Assuming a function extract_license_plate_number exists to extract the plate number
    vehicle_license_plate = extract_license_plate_number(predictions)
    return vehicle_license_plate in authorized_vehicles

# test_function_code --------------------

def test_authorize_vehicle_access():
    print("Testing started.")
    authorized_vehicles = ['ABC123', 'XYZ789', 'TEST456']

    # Testing case 1: Authorized vehicle
    print("Testing case [1/2] started.")
    image_path = 'path/to/authorized_vehicle.jpg'
    assert authorize_vehicle_access(image_path, authorized_vehicles) == True, "Test case [1/2] failed: Expected True for authorized vehicle."

    # Testing case 2: Unauthorized vehicle
    print("Testing case [2/2] started.")
    image_path = 'path/to/unauthorized_vehicle.jpg'
    assert authorize_vehicle_access(image_path, authorized_vehicles) == False, "Test case [2/2] failed: Expected False for unauthorized vehicle."

    print("Testing finished.")

# call_test_function_line --------------------

test_authorize_vehicle_access()