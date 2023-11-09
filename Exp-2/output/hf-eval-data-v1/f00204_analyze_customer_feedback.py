from transformers import pipeline


def analyze_customer_feedback(customer_feedback):
    """
    This function uses the 'cardiffnlp/twitter-xlm-roberta-base-sentiment' model from Hugging Face to analyze the sentiment of a given text.
    The model is a multilingual sentiment analysis model specifically trained on ~198M tweets.
    It can analyze and classify the sentiment as either positive, negative, or neutral.
    
    Args:
    customer_feedback (str): The text to be analyzed.
    
    Returns:
    str: The sentiment of the text.
    """
    model_path = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    sentiment_task = pipeline('sentiment-analysis', model=model_path, tokenizer=model_path)
    sentiment = sentiment_task(customer_feedback)
    return sentiment