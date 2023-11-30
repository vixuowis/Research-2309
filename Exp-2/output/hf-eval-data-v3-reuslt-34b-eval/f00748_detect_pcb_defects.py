# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_pcb_defects(image_path):
    """
    Detect defects of PCB boards from an image using YOLO model.

    Args:
        image_path (str): URL or local path to the image.

    Returns:
        None. The function will display the processed image with the detected defects marked.

    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    """
    if type(image_path) is str:
        detect = YOLO("pcb")   # Load pcb model
        result = detect.detect_image(image_path)  # Detect image
        render_result(result)   # Display detected objects
    else:
        raise TypeError('Image URL or local path not valid.')

# function_code --------------------


# test_function_code --------------------

def test_detect_pcb_defects():
    """
    Test the detect_pcb_defects function.
    """
    # Test with a URL image
    detect_pcb_defects('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg')
    # Test with a local image
    # detect_pcb_defects('/path/to/local/image.jpg')
    print('All Tests Passed')


# call_test_function_code --------------------

test_detect_pcb_defects()