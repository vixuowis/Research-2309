from transformers import pipeline

def classify_article(sequence: str, candidate_labels: list, hypothesis_template: str = 'Ce texte parle de {}.') -> dict:
    '''
    Classify a given article into one of the provided categories using a zero-shot classification model.

    Args:
    sequence (str): The text of the article to classify.
    candidate_labels (list): The potential categories for the article.
    hypothesis_template (str, optional): A template for forming hypotheses for the classifier. Defaults to 'Ce texte parle de {}.'.

    Returns:
    dict: The classification results.
    '''
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    return classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)