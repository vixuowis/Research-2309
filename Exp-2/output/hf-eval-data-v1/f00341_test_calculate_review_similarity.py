def test_calculate_review_similarity():
    """
    This function tests the 'calculate_review_similarity' function.
    It uses two book reviews from online sources.
    The expected output is not known as the function uses a machine learning model, so the test only checks if the output is a float.
    """
    review1 = 'I loved this book, it was fantastic!'
    review2 = 'This book was not to my taste, I found it quite boring.'
    
    # Call the function with the test data
    result = calculate_review_similarity(review1, review2)
    
    # Check if the result is a float
    assert isinstance(result, float), f'Expected float, got {type(result)}'
    
    # Check if the result is in the expected range
    assert -1 <= result <= 1, f'Expected a value between -1 and 1, got {result}'

test_calculate_review_similarity()