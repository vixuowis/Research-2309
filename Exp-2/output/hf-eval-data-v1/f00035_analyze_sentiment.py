from transformers import pipeline


def analyze_sentiment(message):
    """
    This function uses the 'cardiffnlp/twitter-xlm-roberta-base-sentiment' model from Hugging Face to analyze the sentiment of a given message.
    The model is trained on a large dataset of tweets and is designed for sentiment analysis tasks.
    
    Parameters:
    message (str): The message to analyze.
    
    Returns:
    str: The sentiment of the message ('positive', 'negative', or 'neutral').
    """
    # Load the sentiment analysis model
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    
    # Analyze the sentiment of the message
    sentiment_analysis_result = sentiment_task(message)
    
    # Return the sentiment
    return sentiment_analysis_result[0]['label']