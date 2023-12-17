# requirements_file --------------------

!pip install -U pyannote.audio==2.1

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapped_speech(audio_file, access_token):
    """
    Detects overlapped speech segments in an audio file using pyannote.audio.

    :param audio_file: path to the audio file to be processed
    :type audio_file: str
    :param access_token: access token for authentication
    :type access_token: str
    :return: list of tuples (start, end) indicating the segments of overlapped speech
    :rtype: list of tuple(float, float)
    """
    pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
    output = pipeline(audio_file)
    overlapped_segments = [(segment.start, segment.end) for segment in output.get_timeline().support()]
    return overlapped_segments

# test_function_code --------------------

def test_detect_overlapped_speech():
    print("Testing started.")
    access_token = 'test_access_token'
    sample_audio_file = 'test_audio.wav'  # This file should be an actual audio file in the test environment

    # Test case 1: Check if the function returns a list
    print("Testing case [1/1] started.")
    overlapped_segments = detect_overlapped_speech(sample_audio_file, access_token)
    assert isinstance(overlapped_segments, list), "Test case [1/1] failed: The function should return a list"
    print("Testing finished.")

# Run the test function
test_detect_overlapped_speech()