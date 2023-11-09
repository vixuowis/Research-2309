def test_classify_article():
    '''
    Test the classify_article function.
    '''
    sequence = 'L\'équipe de France joue aujourd\'hui au Parc des Princes'
    candidate_labels = ['sport', 'politique', 'santé', 'technologie']
    classification_results = classify_article(sequence, candidate_labels)
    assert isinstance(classification_results, dict), 'The result should be a dictionary.'
    assert 'labels' in classification_results, 'The result dictionary should have a key named labels.'
    assert 'scores' in classification_results, 'The result dictionary should have a key named scores.'
    assert len(classification_results['labels']) == len(candidate_labels), 'The number of labels should be equal to the number of candidate labels.'
    assert len(classification_results['scores']) == len(candidate_labels), 'The number of scores should be equal to the number of candidate labels.'

test_classify_article()