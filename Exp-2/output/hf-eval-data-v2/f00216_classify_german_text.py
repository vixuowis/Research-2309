# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_german_text(sequence: str, candidate_labels: list, hypothesis_template: str = 'In deisem geht es um {}.') -> dict:
    """
    Classify a German text into different categories like crime, tragedy, or theft using a zero-shot classification model.

    Args:
        sequence (str): The input text to be classified.
        candidate_labels (list): A list of candidate labels in German (e.g., 'Verbrechen', 'Tragödie', 'Stehlen').
        hypothesis_template (str, optional): A hypothesis template in German. Defaults to 'In deisem geht es um {}.'.

    Returns:
        dict: The classification result.
    """
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_classify_german_text():
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie'
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    result = classify_german_text(sequence, candidate_labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result dictionary should have a key named labels.'
    assert 'scores' in result, 'The result dictionary should have a key named scores.'

# call_test_function_code --------------------

test_classify_german_text()