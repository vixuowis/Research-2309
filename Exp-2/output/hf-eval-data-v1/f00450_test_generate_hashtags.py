def test_generate_hashtags():
    """
    This function tests the 'generate_hashtags' function by using a sample image URL.
    It asserts that the output is a list and that it is not empty.
    """
    sample_image_url = 'https://example.com/sample.jpg'
    hashtags = generate_hashtags(sample_image_url)

    # Assert that the output is a list
    assert isinstance(hashtags, list), 'Output should be a list.'

    # Assert that the list is not empty
    assert hashtags, 'Output list should not be empty.'

    print('All tests passed.')

test_generate_hashtags()