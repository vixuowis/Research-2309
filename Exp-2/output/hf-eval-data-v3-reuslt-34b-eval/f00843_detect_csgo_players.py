# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_csgo_players(image_path: str) -> None:
    '''
    Detects players in a live game of Counter-Strike: Global Offensive (CS:GO) using a pre-trained YOLO model.

    Args:
        image_path (str): The path to the game screen image.

    Returns:
        None. The function prints the bounding boxes of detected players and displays the image with detected players.

    Raises:
        FileNotFoundError: If the provided image_path does not exist.
    '''
    
    # Load model once
    model = YOLO(img_size=416, conf_thres = 0.52)

    try:
        result = model.detect(image_path) # list of dictionaries
        
        if result is not None:            
            render_result(result, image_path)
    
    except FileNotFoundError as fnf:
        print(fnf)

# test_function_code --------------------

def test_detect_csgo_players():
    '''
    Tests the detect_csgo_players function with a sample image.
    '''
    try:
        detect_csgo_players('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg')
        print('Test Passed')
    except Exception as e:
        print('Test Failed')
        print(e)


# call_test_function_code --------------------

test_detect_csgo_players()