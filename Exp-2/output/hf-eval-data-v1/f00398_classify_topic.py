from transformers import pipeline

def classify_topic(sentence):
    '''
    This function takes a sentence as input and classifies the topic of the sentence among 'technology', 'literature', and 'science'.
    It uses the 'cross-encoder/nli-deberta-v3-xsmall' model from the transformers library for zero-shot classification.
    
    Args:
    sentence (str): The sentence to be classified.
    
    Returns:
    str: The classified topic of the sentence.
    '''
    # Create a zero-shot classification model
    classifier = pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-xsmall')
    # Define the candidate labels
    candidate_labels = ['technology', 'literature', 'science']
    # Classify the sentence
    result = classifier(sentence, candidate_labels)
    # Return the label with the highest score
    return result['labels'][0]