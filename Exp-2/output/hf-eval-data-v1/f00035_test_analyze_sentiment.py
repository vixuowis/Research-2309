def test_analyze_sentiment():
    """
    This function tests the 'analyze_sentiment' function with some example messages.
    """
    # Define some example messages and their expected sentiments
    example_messages = [
        ('I love this product!', 'positive'),
        ('I hate this product!', 'negative'),
        ('This product is okay.', 'neutral'),
    ]
    
    # Test the 'analyze_sentiment' function with the example messages
    for message, expected_sentiment in example_messages:
        assert analyze_sentiment(message) == expected_sentiment, f'For message: {message}, expected: {expected_sentiment} but got different sentiment.'

test_analyze_sentiment()