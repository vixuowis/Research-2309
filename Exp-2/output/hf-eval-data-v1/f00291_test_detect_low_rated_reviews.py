def test_detect_low_rated_reviews():
    """
    This function tests the detect_low_rated_reviews function by using a sample review text.
    """
    review_text = 'I hate this product!'
    assert detect_low_rated_reviews(review_text) == 'Low-rated product review detected'
    
    review_text = 'I love this product!'
    assert detect_low_rated_reviews(review_text) == 'Review is not low-rated'

test_detect_low_rated_reviews()