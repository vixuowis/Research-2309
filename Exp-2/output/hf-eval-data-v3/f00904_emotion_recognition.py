# function_import --------------------

from transformers import pipeline

# function_code --------------------

def emotion_recognition(audio_file_path: str, top_k: int = 1):
    """
    Function to recognize emotion from an audio file using a pre-trained model.

    Args:
        audio_file_path (str): Path to the audio file.
        top_k (int, optional): Number of top predictions to return. Defaults to 1.

    Returns:
        list: List of dictionaries containing 'label' and 'score' of the predicted emotions.
    """
    emotion_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')
    emotion_label = emotion_classifier(audio_file_path, top_k=top_k)
    return emotion_label

# test_function_code --------------------

def test_emotion_recognition():
    """
    Function to test the emotion_recognition function.
    """
    # Test case: Positive case with a valid audio file
    emotion_label = emotion_recognition('valid_audio_file.wav')
    assert isinstance(emotion_label, list), 'The output should be a list.'
    assert 'label' in emotion_label[0], 'Each item in the list should be a dictionary with a label key.'
    assert 'score' in emotion_label[0], 'Each item in the list should be a dictionary with a score key.'

    # Test case: Check the top_k parameter
    emotion_labels = emotion_recognition('valid_audio_file.wav', top_k=5)
    assert len(emotion_labels) == 5, 'The output list should contain 5 items when top_k is set to 5.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_emotion_recognition()