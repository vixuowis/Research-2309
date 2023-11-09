def test_detect_emotion():
    """
    Tests the detect_emotion function.
    """
    test_text = 'I love this!'
    expected_output = [{'label': 'joy', 'score': 0.75}]
    assert detect_emotion(test_text)[0]['label'] == expected_output[0]['label']
    assert abs(detect_emotion(test_text)[0]['score'] - expected_output[0]['score']) < 0.1

test_detect_emotion()