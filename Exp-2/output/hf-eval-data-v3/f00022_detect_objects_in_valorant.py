# function_import --------------------

from ultralytics.yolov5 import YOLO, render_result

# function_code --------------------

def detect_objects_in_valorant(game_frame):
    '''
    Detects objects in a Valorant game frame using the YOLO object detection model.
    
    Args:
        game_frame (str): The URL or local path of the game frame image.
    
    Returns:
        A render object that visualizes the detected objects in the game frame.
    
    Raises:
        ValueError: If the game_frame is not a valid URL or local path.
    '''
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(game_frame)
    render = render_result(model=model, image=game_frame, result=results[0])
    return render

# test_function_code --------------------

def test_detect_objects_in_valorant():
    '''
    Tests the detect_objects_in_valorant function with various test cases.
    '''
    # Test with a valid game frame image URL
    game_frame = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    render = detect_objects_in_valorant(game_frame)
    assert isinstance(render, type(render_result())), 'Test case 1 failed'
    
    # Test with a valid local game frame image path
    game_frame = './test_images/game_frame.jpg'
    render = detect_objects_in_valorant(game_frame)
    assert isinstance(render, type(render_result())), 'Test case 2 failed'
    
    # Test with an invalid game frame image URL
    game_frame = 'https://invalid-url.com/game_frame.jpg'
    try:
        render = detect_objects_in_valorant(game_frame)
    except ValueError:
        pass
    else:
        assert False, 'Test case 3 failed'
    
    return 'All tests passed'

# call_test_function_code --------------------

test_detect_objects_in_valorant()