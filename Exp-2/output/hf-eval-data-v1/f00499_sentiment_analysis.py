from transformers import pipeline


def sentiment_analysis(text):
    """
    This function uses the 'valhalla/distilbart-mnli-12-6' model from the transformers library to perform zero-shot classification.
    It classifies the sentiment of the input text as either 'positive' or 'negative'.
    
    Args:
    text (str): The text to be classified.
    
    Returns:
    dict: The classification results.
    """
    # Create an instance of the zero-shot classification model
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    
    # Classify the sentiment of the text
    result = nlp(text, ['positive', 'negative'])
    
    return result