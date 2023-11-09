def test_analyze_sentiment():
    # Test the analyze_sentiment function with some example reviews
    # Note: The function should return a sentiment analysis result, not a strict number
    # Therefore, we will not use assert to compare the result with a specific number
    # Instead, we will check if the result is in the expected format
    example_reviews = ['Este producto es increíble', 'No me gusta este producto', 'Este producto es normal']
    for review in example_reviews:
        result = analyze_sentiment(review)
        assert isinstance(result, list) and isinstance(result[0], dict) and 'label' in result[0] and 'score' in result[0]

test_analyze_sentiment()