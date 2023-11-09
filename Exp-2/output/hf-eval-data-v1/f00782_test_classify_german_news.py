def test_classify_german_news():
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen Kolonie'
    result = classify_german_news(sequence)
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'scores' in result
    assert len(result['labels']) == len(result['scores'])

test_classify_german_news()