# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_voice_activity(audio_file_path):
    """
    Detects voice activity in an audio file using the FSMN-VAD model from Hugging Face Transformers library.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        A list of tuples where each tuple represents a segment of the audio file. The first element of the tuple is the start time of the segment, the second element is the end time, and the third element is a boolean indicating whether voice activity was detected in this segment.
    """
    voice_activity_detector = pipeline('voice-activity-detection', model='funasr/FSMN-VAD')
    voice_activity = voice_activity_detector(audio_file_path)
    return voice_activity

# test_function_code --------------------

def test_detect_voice_activity():
    """
    Tests the detect_voice_activity function by analyzing a sample audio file and checking the output.
    """
    sample_audio_file_path = 'path_to_sample_audio_file'
    voice_activity = detect_voice_activity(sample_audio_file_path)
    assert isinstance(voice_activity, list), 'The output should be a list.'
    assert all(isinstance(segment, tuple) and len(segment) == 3 for segment in voice_activity), 'Each segment should be a tuple with three elements.'
    assert all(isinstance(segment[0], float) and isinstance(segment[1], float) and isinstance(segment[2], bool) for segment in voice_activity), 'Each segment should contain two floats and a boolean.'

# call_test_function_code --------------------

test_detect_voice_activity()