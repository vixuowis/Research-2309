def test_analyze_user_review():
    '''
    This function tests the 'analyze_user_review' function.
    It uses a sample review and checks if the returned sentiment is one of the expected values (positive, negative, or neutral).
    '''
    sample_review = 'Reseña del usuario aquí...'
    sentiment = analyze_user_review(sample_review)
    assert sentiment in ['POS', 'NEG', 'NEU'], f'Unexpected sentiment: {sentiment}'

test_analyze_user_review()