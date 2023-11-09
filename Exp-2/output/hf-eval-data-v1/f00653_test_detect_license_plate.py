def test_detect_license_plate():
    '''
    This function tests the detect_license_plate function.
    
    It uses a sample image from the keremberke/license-plate-object-detection dataset and checks if the function returns results with the expected structure.
    '''
    # Define the path to the sample image
    img_path = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
    
    # Call the function with the sample image
    results = detect_license_plate(img_path)
    
    # Check if the results have the expected structure
    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'boxes' in results, 'The result should contain bounding boxes.'
    assert 'scores' in results, 'The result should contain scores.'
    assert 'categories' in results, 'The result should contain categories.'
    
    # Check if the bounding boxes, scores, and categories are lists
    assert isinstance(results['boxes'], list), 'The bounding boxes should be a list.'
    assert isinstance(results['scores'], list), 'The scores should be a list.'
    assert isinstance(results['categories'], list), 'The categories should be a list.'
    
    # Check if the lists are not empty
    assert results['boxes'], 'The list of bounding boxes should not be empty.'
    assert results['scores'], 'The list of scores should not be empty.'
    assert results['categories'], 'The list of categories should not be empty.'