def test_classify_houseplant():
    '''
    Test the classify_houseplant function.
    '''
    # Define a test image URL
    url = 'https://example.com/houseplant_image.jpg'

    # Call the function with the test image URL
    result = classify_houseplant(url)

    # Assert that the result is a string (the predicted houseplant type)
    assert isinstance(result, str), 'The result should be a string.'

    # Assert that the result is not empty
    assert result != '', 'The result should not be empty.'

test_classify_houseplant()