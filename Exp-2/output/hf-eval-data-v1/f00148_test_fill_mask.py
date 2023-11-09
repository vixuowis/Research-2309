def test_fill_mask():
    # Test the fill_mask function with a sentence
    sentence = 'The weather today is [MASK] than yesterday.'
    result = fill_mask(sentence)
    # Assert that the result is not None
    assert result is not None
    # Assert that the result is a list
    assert isinstance(result, list)
    # Assert that the list is not empty
    assert len(result) > 0

test_fill_mask()