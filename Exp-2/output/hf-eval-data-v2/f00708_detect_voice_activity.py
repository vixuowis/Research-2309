# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_voice_activity(audio_file):
    """
    This function detects active speech in an audio file using a pretrained voice activity detection pipeline from pyannote.audio.

    Args:
        audio_file (str): Path to the audio file (.wav format).

    Returns:
        A list of tuples, where each tuple represents an active speech segment and contains the start and end times of the segment.
    """
    pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection')
    output = pipeline(audio_file)
    active_speech_segments = []
    for speech in output.get_timeline().support():
        # Active speech between speech.start and speech.end
        active_speech_segments.append((speech.start, speech.end))
    return active_speech_segments

# test_function_code --------------------

def test_detect_voice_activity():
    """
    This function tests the detect_voice_activity function by using a sample audio file.
    The test will pass if the function returns a list of tuples, where each tuple represents an active speech segment.
    """
    sample_audio_file = 'sample.wav'
    result = detect_voice_activity(sample_audio_file)
    assert isinstance(result, list), 'The result should be a list.'
    for segment in result:
        assert isinstance(segment, tuple), 'Each segment should be a tuple.'
        assert len(segment) == 2, 'Each segment should contain start and end times.'
        assert segment[0] < segment[1], 'The start time should be less than the end time.'

# call_test_function_code --------------------

test_detect_voice_activity()