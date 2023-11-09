def test_classify_review():
    review = 'Algún día iré a ver el mundo'
    categories = ['viaje', 'cocina', 'danza']
    result = classify_review(review, categories)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'labels' in result, 'The result should contain labels.'
    assert 'scores' in result, 'The result should contain scores.'
    assert len(result['labels']) == len(categories), 'The number of labels should be equal to the number of categories.'
    assert len(result['scores']) == len(categories), 'The number of scores should be equal to the number of categories.'

test_classify_review()