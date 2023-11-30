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
    
    # Initialize the classifier.
    # The default model is fine for this use case and can be loaded in under two minutes.
    classifier = pipeline('sentiment-analysis')

    # Compute the hypotheses.
    hypothesis_list = [hypothesis_template.format(label) for label in candidate_labels]
    
    # Classify each hypothesis.
    classification_result = {label: score['label'] for (label, score) in zip(candidate_labels, classifier(hypothesis_list))}
    
    # Compute the average confidence score over candidate hypotheses. 
    # Note that this is NOT a standard sentiment score! It ranges from -1 to +1, where positive scores mean 'confidently in favor of the label'. Negative scores indicate 'confidently against it'.
    avg_classification_score = sum([classification_result[label] for label in candidate_labels]) / len(candidate_labels)
    
    # Compute a sentiment score. This ranges from -1 to +1, where positive scores mean 'positive sentiment' and negative scores indicate 'negative sentiment'.
    sentiment_score = (avg_classification_score - 0.5) * 2
    
    return {'sequence': sequence, 'sentiment_score': sentiment_score}

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