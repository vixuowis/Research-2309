# function_import --------------------

from transformers import pipeline
import librosa

# function_code --------------------

def emotion_recognition(audio_file_path: str, top_k: int = 5):
    """
    Analyze the emotions expressed in a user's recorded message using Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file.
        top_k (int, optional): The number of top predicted emotions to return. Defaults to 5.

    Returns:
        list: A list of dictionaries containing the top predicted emotions and their scores.
    """
    classifier = pipeline('audio-classification', model='superb/hubert-large-superb-er')
    predicted_emotions = classifier(audio_file_path, top_k=top_k)
    return predicted_emotions

# test_function_code --------------------

def test_emotion_recognition():
    """
    Test the emotion_recognition function.
    """
    # Test case: Check the type of the returned result
    result = emotion_recognition('path/to/audio/file.wav')
    assert isinstance(result, list), 'The result should be a list.'
    # Test case: Check the length of the returned result
    result = emotion_recognition('path/to/audio/file.wav', top_k=3)
    assert len(result) == 3, 'The length of the result should be equal to top_k.'
    # Test case: Check the keys in the returned dictionaries
    result = emotion_recognition('path/to/audio/file.wav')
    for emotion in result:
        assert 'score' in emotion, 'Each dictionary should have a score key.'
        assert 'entity' in emotion, 'Each dictionary should have an entity key.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_emotion_recognition()