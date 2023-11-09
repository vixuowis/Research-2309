def test_detect_object_in_image():
    '''
    Function to test detect_object_in_image function.
    '''
    # Define the URL of the image and the text queries
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a dog']

    # Call the function with the defined parameters
    results = detect_object_in_image(url, texts)

    # Assert that the function returns a dictionary
    assert isinstance(results, dict), 'The function should return a dictionary.'

    # Assert that the dictionary contains the expected keys
    expected_keys = ['scores', 'labels', 'boxes']
    for key in expected_keys:
        assert key in results, f'The result dictionary should contain the key {key}.'

    # Assert that the scores are within the expected range
    assert all(0 <= score <= 1 for score in results['scores']), 'All scores should be between 0 and 1.'

    # Call the function with a different set of parameters
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a cat']
    results = detect_object_in_image(url, texts)

    # Assert that the function returns a dictionary
    assert isinstance(results, dict), 'The function should return a dictionary.'

    # Assert that the dictionary contains the expected keys
    for key in expected_keys:
        assert key in results, f'The result dictionary should contain the key {key}.'

    # Assert that the scores are within the expected range
    assert all(0 <= score <= 1 for score in results['scores']), 'All scores should be between 0 and 1.'

# Run the test function
test_detect_object_in_image()