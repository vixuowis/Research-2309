from transformers import pipeline


def analyze_sentiment(text):
    """
    Perform sentiment analysis on the provided text using the 'nlptown/bert-base-multilingual-uncased-sentiment' model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The result of the sentiment analysis. The keys are 'label' and 'score'. 'label' is the predicted sentiment (1-5 stars), and 'score' is the confidence of the prediction.
    """
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_result = sentiment_pipeline(text)
    return sentiment_result[0]