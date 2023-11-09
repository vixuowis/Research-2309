from transformers import pipeline


def analyze_feedback_sentiment(customer_feedback_in_spanish):
    """
    This function uses the Transformers library to analyze the sentiment of customer feedback in Spanish.
    It uses the 'finiteautomata/beto-sentiment-analysis' model, which is a BERT model trained in Spanish.
    The model was trained using the TASS 2020 corpus, making it suitable for analyzing sentiment of customer feedback in Spanish.
    The model will classify feedback into the categories of Positive (POS), Negative (NEG) and Neutral (NEU).
    
    Parameters:
    customer_feedback_in_spanish (str): The customer feedback in Spanish to be analyzed.
    
    Returns:
    dict: The sentiment analysis result.
    """
    feedback_sentiment = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = feedback_sentiment(customer_feedback_in_spanish)
    return sentiment_result