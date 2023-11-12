# function_import --------------------

from transformers import pipeline

# function_code --------------------

def german_text_classification(sequence: str, candidate_labels: list, hypothesis_template: str = 'In deisem geht es um {}.') -> dict:
    '''
    Classify a German text into different categories using zero-shot classification.

    Args:
        sequence (str): The input text in German to be classified.
        candidate_labels (list): A list of candidate labels in German.
        hypothesis_template (str): A hypothesis template in German. Default is 'In deisem geht es um {}.'.

    Returns:
        dict: The classification result.
    '''
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_german_text_classification():
    '''
    Test the function german_text_classification.
    '''
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie'
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    result = german_text_classification(sequence, candidate_labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result should contain labels.'
    assert 'scores' in result, 'The result should contain scores.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_german_text_classification()