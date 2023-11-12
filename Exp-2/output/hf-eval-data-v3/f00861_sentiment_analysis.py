# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(review):
    """
    This function uses the Hugging Face Transformers library to perform sentiment analysis on a given movie review.
    The sentiment analysis model used is 'lvwerra/distilbert-imdb', which is trained on the IMDB dataset.

    Args:
        review (str): The movie review to be analyzed.

    Returns:
        dict: The sentiment prediction. Contains two keys - 'label' and 'score'. 'label' can be 'POSITIVE' or 'NEGATIVE'.
        'score' is a float indicating the confidence of the prediction.
    """
    sentiment_classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    sentiment_prediction = sentiment_classifier(review)
    return sentiment_prediction[0]

# test_function_code --------------------

def test_sentiment_analysis():
    """
    This function tests the sentiment_analysis function with some example movie reviews.
    """
    positive_review = "I absolutely loved this movie! The acting, the storyline, and the cinematography were all outstanding."
    negative_review = "I really didn't like this movie. The plot was predictable and the acting was subpar."

    positive_prediction = sentiment_analysis(positive_review)
    negative_prediction = sentiment_analysis(negative_review)

    assert positive_prediction['label'] == 'POSITIVE', f"Expected 'POSITIVE', but got {positive_prediction['label']}"
    assert negative_prediction['label'] == 'NEGATIVE', f"Expected 'NEGATIVE', but got {negative_prediction['label']}"

    return 'All Tests Passed'

# call_test_function_code --------------------

test_sentiment_analysis()