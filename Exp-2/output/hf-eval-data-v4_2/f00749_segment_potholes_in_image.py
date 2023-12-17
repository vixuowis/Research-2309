# requirements_file --------------------

!pip install -U ultralyticsplus==0.0.23 ultralytics==8.0.21

# function_import --------------------

from ultralyticsplus import YOLO, render_result

# function_code --------------------

def segment_potholes_in_image(image_path):
    """
    Segments potholes in the image using YOLOv8 model.

    Args:
        image_path (str): URL or local path to the image to segment.

    Returns:
        Image: The image with pothole detections and segmentation masks rendered.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the image_path is not a valid URL or local file path.
    """
    model = YOLO('keremberke/yolov8s-pothole-segmentation')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    results = model.predict(image_path)

    if results is None or len(results) == 0:
        raise ValueError('No pothole detection results returned')

    render = render_result(model=model, image=image_path, result=results[0])
    render.show()

    return render

# test_function_code --------------------

def test_segment_potholes_in_image():
    print('Testing started.')

    # Use a test image URL
    test_image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

    # Test case 1: Test with valid image URL
    print('Testing case [1/3] started.')
    try:
        result_image = segment_potholes_in_image(test_image_url)
        assert result_image is not None, 'Test case [1/3] failed: No result image returned.'
    except Exception as e:
        assert False, f'Test case [1/3] failed with exception: {str(e)}'

    # Test case 2: Test with invalid image path
    print('Testing case [2/3] started.')
    invalid_image_path = 'non_existent_image.jpg'
    try:
        segment_potholes_in_image(invalid_image_path)
        assert False, 'Test case [2/3] failed: FileNotFoundError not raised for invalid image path.'
    except FileNotFoundError:
        pass
    except Exception as e:
        assert False, f'Test case [2/3] failed with exception: {str(e)}'

    # Test case 3: Test with invalid input type
    print('Testing case [3/3] started.')
    invalid_input = 123
    try:
        segment_potholes_in_image(invalid_input)
        assert False, 'Test case [3/3] failed: TypeError not raised for invalid input type.'
    except TypeError:
        pass
    except Exception as e:
        assert False, f'Test case [3/3] failed with exception: {str(e)}'

    print('Testing finished.')

# call_test_function_line --------------------

test_segment_potholes_in_image()