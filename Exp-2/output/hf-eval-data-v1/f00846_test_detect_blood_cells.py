def test_detect_blood_cells():
    """
    Tests the detect_blood_cells function.
    """
    # Use a sample image for testing
    sample_image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    try:
        detect_blood_cells(sample_image)
        print('Test passed.')
    except Exception as e:
        print('Test failed.\n', e)