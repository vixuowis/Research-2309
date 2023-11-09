def test_detect_pcb_defects():
    """
    This function tests the detect_pcb_defects function by providing a sample image and checking if the output is not None.
    """
    image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    result = detect_pcb_defects(image)
    assert result is not None, 'Test failed: No result returned.'

test_detect_pcb_defects()