def test_emotion_classifier():
    '''
    This function tests the emotion_classifier function.
    It uses a sample text and checks if the function returns a result.
    '''
    text = 'What a fantastic movie! It was so captivating.'
    result = emotion_classifier(text)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'label' in result[0], 'Each item in the result should have a label.'
    assert 'score' in result[0], 'Each item in the result should have a score.'

test_emotion_classifier()