# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(review):
    """
    This function uses the Hugging Face Transformers library to perform sentiment analysis on a given movie review.
    The model used is 'lvwerra/distilbert-imdb', which is a fine-tuned version of distilbert-base-uncased on the imdb dataset.
    It is used for sentiment analysis on movie reviews and achieves an accuracy of 0.928 on the evaluation set.

    Args:
        review (str): The movie review to be analyzed.

    Returns:
        dict: The sentiment prediction. Contains the label ('POSITIVE' or 'NEGATIVE') and the score.
    """
    sentiment_classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    sentiment_prediction = sentiment_classifier(review)
    return sentiment_prediction

# test_function_code --------------------

def test_sentiment_analysis():
    """
    This function tests the sentiment_analysis function with a positive and a negative review.
    """
    positive_review = 'I absolutely loved this movie! The acting, the storyline, and the cinematography were all outstanding.'
    negative_review = 'I really did not like this movie. The plot was predictable and the acting was subpar.'
    positive_prediction = sentiment_analysis(positive_review)
    negative_prediction = sentiment_analysis(negative_review)
    assert positive_prediction[0]['label'] == 'POSITIVE', 'Positive review incorrectly classified.'
    assert negative_prediction[0]['label'] == 'NEGATIVE', 'Negative review incorrectly classified.'

# call_test_function_code --------------------

test_sentiment_analysis()