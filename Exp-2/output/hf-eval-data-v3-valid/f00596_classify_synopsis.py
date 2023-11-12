# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_synopsis(sequence: str, candidate_labels: list, hypothesis_template: str = 'In deisem geht es um {}') -> dict:
    '''
    Classify a movie synopsis into categories: crime, tragedy, and theft.

    Args:
        sequence (str): The movie synopsis in German.
        candidate_labels (list): A list of candidate labels.
        hypothesis_template (str, optional): A German hypothesis template. Defaults to 'In deisem geht es um {}'.

    Returns:
        dict: The classification result.
    '''
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_classify_synopsis():
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie'
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    hypothesis_template = 'In deisem geht es um {}'
    result = classify_synopsis(sequence, candidate_labels, hypothesis_template)
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'scores' in result
    assert len(result['labels']) == len(candidate_labels)
    assert len(result['scores']) == len(candidate_labels)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_synopsis()