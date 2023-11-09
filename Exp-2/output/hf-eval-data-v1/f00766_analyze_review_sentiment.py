from transformers import pipeline

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a movie review using a pre-trained model from Hugging Face Transformers.

    Args:
        review_text (str): The text of the movie review to analyze.

    Returns:
        str: The sentiment of the review ('positive' or 'negative').

    Raises:
        ValueError: If the review_text is not a string.
    """
    if not isinstance(review_text, str):
        raise ValueError('Review text must be a string.')

    classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    result = classifier(review_text)
    return result[0]['label']