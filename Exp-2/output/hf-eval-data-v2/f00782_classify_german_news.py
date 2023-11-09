# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_german_news(sequence):
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

# test_function_code --------------------

def test_classify_german_news():
    """
    Test the classify_german_news function.
    """
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen Kolonie'
    result = classify_german_news(sequence)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result should contain labels.'
    assert 'scores' in result, 'The result should contain scores.'

# call_test_function_code --------------------

test_classify_german_news()