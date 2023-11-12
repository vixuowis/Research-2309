# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_french_news(sequence: str, candidate_labels: list) -> dict:
    """
    Classify French news into categories using zero-shot classification.

    Args:
        sequence (str): The news article to be classified.
        candidate_labels (list): The list of categories to classify the news into.

    Returns:
        dict: A dictionary containing the categories and their corresponding probabilities.

    Raises:
        OSError: If the model 'BaptisteDoyen/camembert-base-xnli' is not found.
    """
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    hypothesis_template = 'Ce texte parle de {}.'
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_classify_french_news():
    """
    Test the function classify_french_news.
    """
    sequence = 'L\'équipe de France joue aujourd\'hui au Parc des Princes'
    candidate_labels = ['sport', 'politique', 'science']
    result = classify_french_news(sequence, candidate_labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert set(candidate_labels).issubset(result.keys()), 'The result should contain all candidate labels.'
    assert all(isinstance(value, float) for value in result.values()), 'All values in the result should be floats.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_french_news()