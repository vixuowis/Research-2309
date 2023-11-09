import yolov5

# Function to detect abnormal objects in an image
# Parameters:
# img: The image URL or file path
# Returns:
# A list of abnormal objects detected in the image

def detect_abnormal_objects(img):
    # Load the yolov5 model
    model = yolov5.load('fcakyon/yolov5s-v7.0')
    # Set model parameters
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    # Perform object detection on the image
    results = model(img, size=640, augment=True)
    # Get the predictions
    predictions = results.pred[0]
    # Get the bounding boxes, scores, and categories
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    # Identify abnormal objects
    abnormal_objects = []
    for i in range(len(categories)):
        if categories[i] not in ['person', 'bag', 'umbrella']:
            abnormal_objects.append((boxes[i], scores[i], categories[i]))
    return abnormal_objects