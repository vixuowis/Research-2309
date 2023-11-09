from transformers import pipeline


def analyze_sentiment(text):
    """
    This function uses the 'siebert/sentiment-roberta-large-english' model from the Transformers library
    to analyze the sentiment of a given text. The model is a fine-tuned checkpoint of RoBERTa-large
    and is capable of predicting either positive (1) or negative (0) sentiment.
    
    Args:
    text (str): The text to analyze.
    
    Returns:
    dict: The sentiment analysis result.
    """
    sentiment_analysis = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
    result = sentiment_analysis(text)
    return result