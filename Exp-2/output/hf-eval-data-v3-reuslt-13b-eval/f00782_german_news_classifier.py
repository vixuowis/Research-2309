# function_import --------------------

from transformers import pipeline

# function_code --------------------

def german_news_classifier(sequence: str, candidate_labels: list, hypothesis_template: str = 'In diesem Text geht es um {}.') -> dict:
    '''
    Classify German news articles into categories like crime, tragedy, and theft using a zero-shot classification model.

    Args:
        sequence (str): The German news article to be classified.
        candidate_labels (list): The list of categories to classify the article into. For example: ['Verbrechen', 'Tragödie', 'Stehlen']
        hypothesis_template (str): The template for the classification task. Default is 'In diesem Text geht es um {}.'

    Returns:
        dict: The classification result.
    '''    

    model = pipeline('zero-shot-classification')
    return model(sequence, candidate_labels, hypothesis_template=hypothesis_template)

# test_function_code --------------------

def test_german_news_classifier():
    '''
    Test the german_news_classifier function.
    '''
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen Kolonie'
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    result = german_news_classifier(sequence, candidate_labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result dictionary should have a key named labels.'
    assert 'scores' in result, 'The result dictionary should have a key named scores.'
    assert len(result['labels']) == len(candidate_labels), 'The number of labels in the result should be equal to the number of candidate labels.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_german_news_classifier()