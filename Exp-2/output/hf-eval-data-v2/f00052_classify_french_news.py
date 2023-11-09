# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_french_news(sequence: str, candidate_labels: list, hypothesis_template: str = 'Ce texte parle de {}.'):
    """
    Classify French news articles into categories using zero-shot classification.

    Args:
        sequence (str): The text (news article) to categorize.
        candidate_labels (list): The categories to classify the text into, such as sports, politics, and science.
        hypothesis_template (str, optional): A template that identifies the categories. Defaults to 'Ce texte parle de {}.'.

    Returns:
        dict: The result with probabilities associated with each category.
    """
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_classify_french_news():
    """
    Test the classify_french_news function.
    """
    sequence = 'L\'équipe de France joue aujourd\'hui au Parc des Princes'
    candidate_labels = ['sport', 'politique', 'science']
    result = classify_french_news(sequence, candidate_labels)
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'scores' in result
    assert len(result['labels']) == len(candidate_labels)
    assert len(result['scores']) == len(candidate_labels)

# call_test_function_code --------------------

test_classify_french_news()