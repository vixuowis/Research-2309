def test_predict_emotion_hubert():
    # Test file path
    test_file_path = 'male-crying.mp3'
    # Call the function with the test file
    result = predict_emotion_hubert(test_file_path)
    # Check the result is a list
    assert isinstance(result, list), 'Result should be a list.'
    # Check the length of the result
    assert len(result) > 0, 'Result should not be empty.'
    # Check the elements of the result
    for res in result:
        assert 'emo' in res, 'Each element should have an emotion label.'
        assert 'score' in res, 'Each element should have a score.'
        assert isinstance(res['emo'], str), 'Emotion label should be a string.'
        assert isinstance(res['score'], float), 'Score should be a float.'

test_predict_emotion_hubert()