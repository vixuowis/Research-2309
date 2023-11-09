def test_analyze_movie_review_sentiment():
    review = 'I absolutely loved this movie! The acting, the storyline, and the cinematography were all outstanding.'
    prediction = analyze_movie_review_sentiment(review)
    assert prediction['label'] in ['POSITIVE', 'NEGATIVE']
    assert 0 <= prediction['score'] <= 1

test_analyze_movie_review_sentiment()