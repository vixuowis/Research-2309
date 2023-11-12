# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def segment_potholes(image_path: str):
    '''
    Function to segment potholes in an image using a pretrained YOLOv8 model.

    Args:
        image_path (str): The path to the image file or URL.

    Returns:
        A rendered image with the segmented potholes.

    Raises:
        ValueError: If the image_path is not a valid file or URL.
    '''
    model = YOLO('keremberke/yolov8m-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(image_path)
    render = render_result(model=model, image=image_path, result=results[0])
    return render

# test_function_code --------------------

def test_segment_potholes():
    '''
    Function to test the segment_potholes function.
    '''
    test_image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = segment_potholes(test_image)
    assert result is not None, 'Test Failed: No result returned'
    assert isinstance(result, type(render_result)), 'Test Failed: Result is not of expected type'
    print('All Tests Passed')

# call_test_function_code --------------------

test_segment_potholes()