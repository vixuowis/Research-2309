# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapping_speech(audio_file: str, access_token: str):
    """
    Detects overlapping speech in an audio file using the pyannote.audio framework.

    Args:
        audio_file (str): Path to the audio file.
        access_token (str): Access token for the pretrained model.

    Returns:
        list: A list of tuples, each containing the start and end times of overlapping speech segments.
    """
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    output = pipeline(audio_file)
    overlapping_speech_segments = []
    for speech in output.get_timeline().support():
        start_time, end_time = speech.start, speech.end
        overlapping_speech_segments.append((start_time, end_time))
    return overlapping_speech_segments

# test_function_code --------------------

def test_detect_overlapping_speech():
    """
    Tests the detect_overlapping_speech function with a sample audio file.
    """
    # Replace with a valid access token and audio file for testing
    access_token = 'ACCESS_TOKEN_GOES_HERE'
    audio_file = 'test_audio.wav'
    overlapping_speech_segments = detect_overlapping_speech(audio_file, access_token)
    assert isinstance(overlapping_speech_segments, list)
    assert all(isinstance(segment, tuple) and len(segment) == 2 for segment in overlapping_speech_segments)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_overlapping_speech()