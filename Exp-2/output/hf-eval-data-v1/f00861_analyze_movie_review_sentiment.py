from transformers import pipeline


def analyze_movie_review_sentiment(review: str) -> dict:
    """
    Analyze the sentiment of a movie review using a pre-trained model from Hugging Face Transformers.

    Args:
        review (str): The movie review to analyze.

    Returns:
        dict: The sentiment prediction. Contains two keys: 'label' and 'score'. 'label' is either 'POSITIVE' or 'NEGATIVE', and 'score' is a float between 0 and 1 indicating the confidence of the prediction.
    """
    sentiment_classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    sentiment_prediction = sentiment_classifier(review)
    return sentiment_prediction[0]