def test_detect_objects_in_images():
    '''
    This function tests the detect_objects_in_images function.
    '''
    # Define the URL of the test image and the texts representing the objects of interest
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = [['a photo of a living room', 'a photo of a kitchen', 'a photo of a bedroom', 'a photo of a bathroom']]

    # Call the detect_objects_in_images function
    results = detect_objects_in_images(url, texts)

    # Assert that the results are not None
    assert results is not None

    # Assert that the results are a dictionary
    assert isinstance(results, dict)

    # Assert that the results contain the 'labels' key
    assert 'labels' in results

test_detect_objects_in_images()