def test_is_good_time_to_visit():
    '''
    This function tests the is_good_time_to_visit function.
    It uses a sample image and checks if the output is a boolean.
    '''
    image_path = 'path_to_test_image.jpg'
    result = is_good_time_to_visit(image_path)
    assert isinstance(result, bool), 'The function should return a boolean.'