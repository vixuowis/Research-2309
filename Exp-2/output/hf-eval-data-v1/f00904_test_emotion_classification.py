def test_emotion_classification():
    """
    Test function for emotion_classification function.
    """
    # Test with a sample audio file
    sample_audio_file = 'sample.wav'
    result = emotion_classification(sample_audio_file, top_k=1)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'label' in result[0], 'Each item in the list should be a dictionary with a label.'
    assert 'score' in result[0], 'Each item in the list should be a dictionary with a score.'

test_emotion_classification()