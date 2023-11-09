from transformers import pipeline


def analyze_customer_sentiment(customer_message):
    """
    This function uses the Hugging Face Transformers library to analyze the sentiment of a customer message.
    It uses a pre-trained BERT-based model for sentiment analysis on Spanish texts.
    The sentiment analysis model produces classification labels such as POS (positive), NEG (negative), or NEU (neutral).
    
    Args:
        customer_message (str): The customer's message to be analyzed.
    
    Returns:
        str: The sentiment of the customer's message (POS, NEG, NEU).
    """
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    result = sentiment_analyzer(customer_message)
    return result[0]['label']