# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapping_speech(audio_file, access_token):
    """
    Detects periods of overlapping speech in an audio file using the pyannote.audio framework.

    Args:
        audio_file (str): Path to the audio file.
        access_token (str): Access token for the pyannote.audio API.

    Returns:
        A list of tuples, where each tuple represents a period of overlapping speech and contains the start and end times (in seconds).
    """
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    output = pipeline(audio_file)
    overlapping_periods = []
    for speech in output.get_timeline().support():
        start_time, end_time = speech.start, speech.end
        overlapping_periods.append((start_time, end_time))
    return overlapping_periods

# test_function_code --------------------

def test_detect_overlapping_speech():
    """
    Tests the detect_overlapping_speech function by running it on a sample audio file and checking the output.
    """
    # Replace with your actual access token
    access_token = 'ACCESS_TOKEN_GOES_HERE'
    # Replace with the path to your sample audio file
    audio_file = 'sample_audio.wav'
    overlapping_periods = detect_overlapping_speech(audio_file, access_token)
    # Check that the function returns a list
    assert isinstance(overlapping_periods, list)
    # Check that each item in the list is a tuple
    for period in overlapping_periods:
        assert isinstance(period, tuple)
        # Check that each tuple contains two items (start and end times)
        assert len(period) == 2

# call_test_function_code --------------------

test_detect_overlapping_speech()