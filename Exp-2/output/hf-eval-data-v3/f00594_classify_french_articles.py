# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_french_articles(sequence: str, candidate_labels: list, hypothesis_template: str = 'Ce texte parle de {}.') -> dict:
    '''
    Classify French articles into categories using zero-shot classification.

    Args:
        sequence (str): The text of the article to classify.
        candidate_labels (list): The list of potential categories for the article.
        hypothesis_template (str, optional): A template for forming hypotheses for the classifier. Defaults to 'Ce texte parle de {}.'.

    Returns:
        dict: The classification results, including the labels and their corresponding scores.
    '''
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    category_predictions = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return category_predictions

# test_function_code --------------------

def test_classify_french_articles():
    '''
    Test the classify_french_articles function.
    '''
    sequence = "L'équipe de France joue aujourd'hui au Parc des Princes"
    candidate_labels = ['sport', 'politique', 'santé', 'technologie']
    result = classify_french_articles(sequence, candidate_labels)
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'scores' in result
    assert len(result['labels']) == len(candidate_labels)
    assert len(result['scores']) == len(candidate_labels)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_french_articles()