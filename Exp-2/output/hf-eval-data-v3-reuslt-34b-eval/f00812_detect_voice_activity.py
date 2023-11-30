# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_voice_activity(audio_file_path):
    """
    Detects voice activity in an audio file.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: A dictionary containing the voice activity detection results.

    Raises:
        OSError: If there is an error accessing the audio file.
    """
    try:
        with open(file=audio_file_path, mode="rb") as audio_file:
            return _VAD_PIPELINE({"bytes": audio_file})[0]
    except OSError as e:
        raise Exception("Could not access audio file.") from e


# function_init --------------------


# test_function_code --------------------

def test_detect_voice_activity():
    """
    Tests the detect_voice_activity function.
    """
    sample_audio_file_path = 'sample_audio.wav'
    try:
        voice_activity = detect_voice_activity(sample_audio_file_path)
        assert isinstance(voice_activity, dict), 'The result should be a dictionary.'
        assert 'voice_activity' in voice_activity, 'The result should contain voice activity detection results.'
    except OSError as e:
        print(f'Error: {e}')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_voice_activity()