def test_analyze_movie_review():
    """
    Test the analyze_movie_review function.

    This function asserts that the analyze_movie_review function correctly classifies
    a positive and a negative review.
    """
    positive_review = 'The movie Inception is an exceptional piece of cinematic art. The storyline is thought-provoking and keeps the audience engaged till the end. The special effects are breathtaking and complement the plot perfectly.'
    negative_review = 'The movie Inception was a huge disappointment. The plot was confusing and the special effects were underwhelming.'

    positive_result = analyze_movie_review(positive_review)
    negative_result = analyze_movie_review(negative_review)

    assert positive_result['labels'][0] == 'positive', 'Positive review incorrectly classified.'
    assert negative_result['labels'][0] == 'negative', 'Negative review incorrectly classified.'

test_analyze_movie_review()