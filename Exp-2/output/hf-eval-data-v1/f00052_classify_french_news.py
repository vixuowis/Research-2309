from transformers import pipeline

def classify_french_news(sequence):
    '''
    This function classifies a given French news article into one of the categories: sports, politics, or science.
    It uses the Hugging Face Transformers library and a pre-trained model for French zero-shot classification.
    
    Parameters:
    sequence (str): The news article to be classified.
    
    Returns:
    dict: A dictionary with the categories as keys and their respective probabilities as values.
    '''
    # Load the pre-trained model
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    
    # Define the candidate labels
    candidate_labels = ['sport', 'politique', 'science']
    
    # Define the hypothesis template
    hypothesis_template = 'Ce texte parle de {}.'
    
    # Classify the sequence
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    
    return result