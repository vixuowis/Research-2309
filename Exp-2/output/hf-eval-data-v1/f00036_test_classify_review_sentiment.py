def test_classify_review_sentiment():
    """
    This function tests the classify_review_sentiment function.
    It uses a sample review and asserts that the output is either 'POSITIVE' or 'NEGATIVE'.
    """
    # Define a sample review
    review = 'I really enjoyed this book!'
    
    # Call the function with the sample review
    sentiment = classify_review_sentiment(review)
    
    # Assert that the output is either 'POSITIVE' or 'NEGATIVE'
    assert sentiment in ['POSITIVE', 'NEGATIVE'], f'Expected POSITIVE or NEGATIVE, but got {sentiment}'
    
    print('All tests passed.')

# Run the test function
test_classify_review_sentiment()