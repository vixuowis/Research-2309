def test_predict_punctuation():
    """
    Tests the predict_punctuation function.
    """
    test_text = 'Hello world This is a test text'
    expected_output = {'Hello': 'No punctuation', 'world': 'No punctuation', 'This': 'No punctuation', 'is': 'No punctuation', 'a': 'No punctuation', 'test': 'No punctuation', 'text': 'No punctuation'}
    assert predict_punctuation(test_text) == expected_output