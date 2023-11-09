from transformers import pipeline

def analyze_sentiment(review):
    """
    Analyze the sentiment of a review using the FinBERT model.

    Args:
        review (str): The review to analyze.

    Returns:
        str: The sentiment of the review ('positive', 'negative', or 'neutral').
    """
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    result = classifier(review)
    return result[0]['label']