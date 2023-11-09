def test_detect_blood_cells():
    """
    This function tests the detect_blood_cells function by using a sample image.
    """
    # Define the path to the sample image
    image_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    
    # Call the detect_blood_cells function
    result = detect_blood_cells(image_path)
    
    # Assert that the result is not None
    assert result is not None, 'The function did not return a result.'
    
    # Assert that the result is an instance of the expected class
    assert isinstance(result, type(render_result)), 'The function did not return an instance of the expected class.'

test_detect_blood_cells()