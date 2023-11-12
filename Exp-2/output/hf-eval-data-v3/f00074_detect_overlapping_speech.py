# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapping_speech(audio_file: str, access_token: str) -> list:
    """
    Detects when two or more speakers are speaking at the same time in an audio file.

    Args:
        audio_file (str): The path to the audio file.
        access_token (str): The access token for the pretrained model.

    Returns:
        list: A list of tuples where each tuple represents a segment of overlapping speech. Each tuple contains the start and end times of the overlapping speech segment.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the access token is invalid.
    """
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    output = pipeline(audio_file)
    overlapping_speech_segments = []
    for speech in output.get_timeline().support():
        overlapping_speech_segments.append((speech.start, speech.end))
    return overlapping_speech_segments

# test_function_code --------------------

def test_detect_overlapping_speech():
    """Tests the detect_overlapping_speech function."""
    # Test with a valid audio file and access token
    overlapping_speech_segments = detect_overlapping_speech('valid_audio.wav', 'valid_access_token')
    assert isinstance(overlapping_speech_segments, list)
    assert all(isinstance(segment, tuple) and len(segment) == 2 for segment in overlapping_speech_segments)

    # Test with an invalid audio file
    try:
        detect_overlapping_speech('invalid_audio.wav', 'valid_access_token')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    # Test with an invalid access token
    try:
        detect_overlapping_speech('valid_audio.wav', 'invalid_access_token')
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_overlapping_speech()