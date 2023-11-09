def test_image_content_filter():
    """
    This function tests the image_content_filter function by using a test image.
    """
    # Test image URL
    test_image_url = 'https://example.com/test_image.jpg'
    # Expected result
    expected_result = 'Passed'
    # Get the actual result
    actual_result = image_content_filter(test_image_url)
    # Assert that the actual result is not 'Filtered'
    assert actual_result != 'Filtered', f'For image {test_image_url}, expected {expected_result} but got {actual_result}'

test_image_content_filter()