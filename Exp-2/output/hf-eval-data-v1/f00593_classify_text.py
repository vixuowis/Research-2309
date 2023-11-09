from transformers import pipeline

def classify_text(text_message, candidate_labels):
    '''
    This function classifies a given text message into one of the provided categories using a zero-shot classification model.
    
    Parameters:
    text_message (str): The text message to be classified.
    candidate_labels (list): The potential categories for the text message.
    
    Returns:
    dict: A dictionary containing the most likely category and its associated score.
    '''
    # Create a zero-shot text classifier
    classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    
    # Classify the text message
    classification_result = classifier(text_message, candidate_labels)
    
    return classification_result