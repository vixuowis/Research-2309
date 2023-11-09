def test_detect_objects():
    '''
    Test the detect_objects function.
    '''
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = detect_objects(image_url)
    assert isinstance(result, str), 'The result should be a string.'
    assert 'Detected' in result, 'The result should contain the word Detected.'
    assert 'location' in result, 'The result should contain the word location.'

test_detect_objects()