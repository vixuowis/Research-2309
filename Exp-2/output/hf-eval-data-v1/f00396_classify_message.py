from transformers import pipeline

def classify_message(message_text):
    '''
    This function classifies a message as either 'safe' or 'inappropriate' using the 'valhalla/distilbart-mnli-12-3' model from the transformers library.
    
    Parameters:
    message_text (str): The message to be classified.
    
    Returns:
    str: The classification of the message ('safe' or 'inappropriate').
    '''
    classifier = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-3')
    message_classification = classifier(message_text, candidate_labels=['safe', 'inappropriate'])
    if message_classification['labels'][0] == 'inappropriate':
        return 'Warning: Inappropriate message detected.'
    else:
        return 'Safe message.'