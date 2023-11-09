# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapping_speech(audio_file, access_token):
    """
    Detects overlapping speech in an audio file using the pyannote.audio framework.

    Args:
        audio_file (str): Path to the audio file.
        access_token (str): Access token for the pretrained model.

    Returns:
        list: A list of tuples where each tuple represents a segment of overlapping speech. Each tuple contains the start and end times of the segment.
    """
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    output = pipeline(audio_file)
    overlapping_speech_segments = []
    for speech in output.get_timeline().support():
        overlapping_speech_segments.append((speech.start, speech.end))
    return overlapping_speech_segments

# test_function_code --------------------

def test_detect_overlapping_speech():
    """
    Tests the detect_overlapping_speech function by using a sample audio file.
    """
    audio_file = 'sample_audio.wav'
    access_token = 'ACCESS_TOKEN_GOES_HERE'
    overlapping_speech_segments = detect_overlapping_speech(audio_file, access_token)
    assert isinstance(overlapping_speech_segments, list), 'The output should be a list.'
    for segment in overlapping_speech_segments:
        assert isinstance(segment, tuple), 'Each segment should be a tuple.'
        assert len(segment) == 2, 'Each segment should contain start and end times.'
        assert segment[0] < segment[1], 'The start time should be less than the end time.'

# call_test_function_code --------------------

test_detect_overlapping_speech()