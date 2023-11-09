def test_classify_article():
    """
    Test the classify_article function.
    """
    sequence_to_classify = 'Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU'
    output = classify_article(sequence_to_classify)
    assert isinstance(output, dict)
    assert 'labels' in output
    assert 'scores' in output
    assert len(output['labels']) == len(output['scores'])
    assert output['labels'][0] in ['politics', 'economy', 'entertainment', 'environment']

test_classify_article()