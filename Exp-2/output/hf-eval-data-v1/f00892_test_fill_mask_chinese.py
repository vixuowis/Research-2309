def test_fill_mask_chinese():
    text = '我们很高兴与您合作，希望我们的<mask>能为您带来便利。'
    result = fill_mask_chinese(text)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'score' in result[0], 'Each item in the list should be a dictionary with a score.'
    assert 'token_str' in result[0], 'Each item in the list should be a dictionary with a token string.'

test_fill_mask_chinese()