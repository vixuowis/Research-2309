# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_voice_activity(audio_file):
    """
    Detects active speech in an audio file using a pretrained voice activity detection pipeline.

    Args:
        audio_file (str): Path to the audio file (.wav format).

    Returns:
        list: A list of tuples where each tuple represents the start and end times of an active speech segment.
    """
    pipeline = Pipeline.from_pretrained('pyannote/voice-activity-detection')
    output = pipeline(audio_file)
    speech_segments = []
    for speech in output.get_timeline().support():
        # Active speech between speech.start and speech.end
        speech_segments.append((speech.start, speech.end))
    return speech_segments

# test_function_code --------------------

def test_detect_voice_activity():
    """
    Tests the detect_voice_activity function with a sample audio file.
    """
    # Note: Replace 'sample.wav' with the path to a real audio file for testing
    speech_segments = detect_voice_activity('sample.wav')
    assert isinstance(speech_segments, list), 'Output should be a list.'
    assert all(isinstance(segment, tuple) and len(segment) == 2 for segment in speech_segments), 'Each segment should be a tuple of two elements.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_voice_activity()