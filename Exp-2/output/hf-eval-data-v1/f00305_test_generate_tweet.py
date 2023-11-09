def test_generate_tweet():
    """
    This function tests the 'generate_tweet' function.
    """
    # Define the test topic
    test_topic = 'The Future of AI in Education'
    
    # Generate a tweet on the test topic
    test_tweet = generate_tweet(test_topic)
    
    # Assert that the generated tweet is a string
    assert isinstance(test_tweet, str), 'The output should be a string.'
    
    # Assert that the generated tweet is not empty
    assert len(test_tweet) > 0, 'The output should not be an empty string.'
    
    # Assert that the generated tweet is within the Twitter character limit
    assert len(test_tweet) <= 280, 'The output should not exceed 280 characters.'
    
    print('All tests passed.')

test_generate_tweet()