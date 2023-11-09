def test_classify_emotion():
    """
    Test the 'classify_emotion' function.
    """
    test_text = 'I am so happy today!'
    result = classify_emotion(test_text)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'label' in result[0], 'Each item in the result should have a label.'
    assert 'score' in result[0], 'Each item in the result should have a score.'

test_classify_emotion()