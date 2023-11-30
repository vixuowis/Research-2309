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
    if image_path is None or (not os.path.exists(image_path)):
        raise FileNotFoundError('The given path to the image file is invalid!')
    
    # load yolov5 model
    model = YOLO()

    # detect players in the image
    detections, img, img_name, scores, labels = model.detect(image_path)

    # draw bounding boxes and display image with detected players
    render_result(detections, img, img_name, scores, labels)
        
# function_test --------------------    
'''
if __name__ == '__main__':
    print('Please use detect.ipynb')
'''


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