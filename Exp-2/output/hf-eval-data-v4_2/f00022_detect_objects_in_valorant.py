# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def detect_objects_in_valorant(game_frame_image):
    """
    Detects various objects within a Valorant game frame using YOLO algorithm.

    Args:
        game_frame_image: A string path to the image or a PIL image of the game frame to be analyzed.

    Returns:
        A rendered image with detected objects.

    Raises:
        ValueError: If the game_frame_image is not a valid image.
    """
    model = YOLO('keremberke/yolov8m-valorant-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(game_frame_image)
    render = render_result(model=model, image=game_frame_image, result=results[0])
    return render


# test_function_code --------------------

def test_detect_objects_in_valorant():
    print("Testing started.")
    sample_image = 'path/to/test/image.jpg'

    # Testing case 1: Detecting objects in the sample image
    print("Testing case [1/1] started.")
    render = detect_objects_in_valorant(sample_image)
    assert render is not None, f"Test case [1/1] failed: No rendered image returned"
    print("Testing finished.")


# call_test_function_line --------------------

test_detect_objects_in_valorant()