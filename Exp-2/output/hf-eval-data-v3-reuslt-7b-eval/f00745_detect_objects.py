# function_import --------------------

import yolov5

# function_code --------------------

def detect_objects(image_path):
    """
    Detect objects in an image using the YOLOv5 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the bounding boxes, scores, and categories of the detected objects.
    """

    # Load an image from disk using Pillow
    img = Image.open(image_path)
    
    # Create a list to store the predictions in
    preds_out = []
    
    # Run the model and obtain predictions
    preds = yolov5.run(img, size=640, conf=.25, iou=.45, classes=[0])
    
    # Loop over the preds
    for pred in preds: 
        
        # Store the bounding box coordinates
        x1 = pred["x1"]
        y1 = pred["y1"]
        x2 = pred["x2"]
        y2 = pred["y2"]
    
        # Store the score for the object detection
        pred_score = pred["score"]
        
        # Store the category index of the object detected. 
        pred_class = pred['category']
            
        # Create a dictionary and append to the list, containing the bounding box coordinates, score, and category index.
        preds_out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": pred_score, "category": pred_class})
        
    # Return the predictions as a dictionary
    return {"predictions": preds_out}

# test_function_code --------------------

def test_detect_objects():
    """
    Test the detect_objects function.
    """
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    results = detect_objects(image_path)
    assert isinstance(results, dict)
    assert 'boxes' in results
    assert 'scores' in results
    assert 'categories' in results
    assert isinstance(results['boxes'], list)
    assert isinstance(results['scores'], list)
    assert isinstance(results['categories'], list)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_objects()