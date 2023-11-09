import yolov5

# Function to detect license plate from an image
# The function uses a pre-trained model 'keremberke/yolov5m-license-plate' for license plate detection
# The function takes an image path as input and returns whether the vehicle is authorized or not

def detect_license_plate(img_path):
    # Load the pre-trained model
    model = yolov5.load('keremberke/yolov5m-license-plate')
    # Set the configuration parameters for the model
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    # Apply the model to the input image
    results = model(img_path, size=640)
    # Extract the predictions, bounding boxes, scores, and categories
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    # List of authorized license plates
    authorized_vehicles = ['XYZ123', 'ABC789', 'LMN456']
    # Extract license plate number from image
    vehicle_license_plate = '...'
    # Check if the vehicle is authorized or not
    if vehicle_license_plate in authorized_vehicles:
        return 'Access granted'
    else:
        return 'Access denied'