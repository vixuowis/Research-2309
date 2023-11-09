def test_classify_german_text():
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie'
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    result = classify_german_text(sequence, candidate_labels)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result dictionary should have a key named labels.'
    assert 'scores' in result, 'The result dictionary should have a key named scores.'
    assert len(result['labels']) == len(candidate_labels), 'The number of labels in the result should match the number of candidate labels.'
    assert len(result['scores']) == len(candidate_labels), 'The number of scores in the result should match the number of candidate labels.'

test_classify_german_text()