def test_detect_kitchen_objects():
    '''
    This function tests the detect_kitchen_objects function.
    It uses a sample image from an online source and checks if the function returns the expected output.
    '''
    # Define the image path and score threshold
    image_path = 'https://images.unsplash.com/photo-1516685018646-549198525c42'
    score_threshold = 0.1
    
    # Call the function with the test inputs
    detections = detect_kitchen_objects(image_path, score_threshold)
    
    # Check if the function returns a list
    assert isinstance(detections, list), 'The function should return a list.'
    
    # Check if the list contains tuples
    for detection in detections:
        assert isinstance(detection, tuple), 'Each detection should be a tuple.'
        assert len(detection) == 3, 'Each detection should contain three elements.'
        assert isinstance(detection[0], str), 'The first element of the detection should be a string.'
        assert isinstance(detection[1], float), 'The second element of the detection should be a float.'
        assert isinstance(detection[2], list), 'The third element of the detection should be a list.'
        assert len(detection[2]) == 4, 'The third element of the detection should contain four elements.'
    
    print('All tests passed.')

test_detect_kitchen_objects()