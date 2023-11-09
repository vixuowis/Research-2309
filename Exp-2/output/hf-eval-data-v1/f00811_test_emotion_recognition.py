def test_emotion_recognition():
    """
    Test the emotion_recognition function.
    """
    from datasets import load_dataset
    dataset = load_dataset('anton-l/superb_demo', 'er', split='session1')
    audio_file_path = dataset[0]['file']
    predicted_emotions = emotion_recognition(audio_file_path)
    assert isinstance(predicted_emotions, dict), 'The result should be a dictionary.'
    assert 'label' in predicted_emotions[0], 'Each prediction should have a label.'
    assert 'score' in predicted_emotions[0], 'Each prediction should have a score.'

test_emotion_recognition()