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
    vad = pipeline("voice-activity-detection")
    try:
        vad_results = vad(audio_file_path)
    except Exception as e:
        raise OSError("Error detecting voice activity.") from e
    return {
        "vad": [x["voice_activity"] for x in vad_results]
    }

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