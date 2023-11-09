from transformers import pipeline

def classify_german_news(sequence: str):
    """
    Classify German news articles into categories like crime, tragedy, and theft.
    
    Args:
        sequence (str): The German news article to be classified.
    
    Returns:
        dict: The classification result.
    """
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    hypothesis_template = 'In diesem Text geht es um {}.'
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return result