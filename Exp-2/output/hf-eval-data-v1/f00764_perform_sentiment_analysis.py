from transformers import pipeline


def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on the given text using the 'cardiffnlp/twitter-xlm-roberta-base-sentiment' model.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: The result of the sentiment analysis. The keys are 'label' and 'score'.
    """
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    result = sentiment_task(text)
    return result