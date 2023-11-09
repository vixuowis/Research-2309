from transformers import pipeline


def analyze_user_review(user_review):
    '''
    This function takes a user review as input and returns the sentiment of the review.
    It uses the 'finiteautomata/beto-sentiment-analysis' model from the Transformers library to analyze the sentiment.
    The model is trained on the TASS 2020 corpus and uses the BETO base model specifically for Spanish text.
    
    Args:
    user_review (str): The user review to be analyzed.
    
    Returns:
    str: The sentiment of the review (positive, negative, or neutral).
    '''
    sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')
    sentiment_result = sentiment_analyzer(user_review)
    return sentiment_result[0]['label']