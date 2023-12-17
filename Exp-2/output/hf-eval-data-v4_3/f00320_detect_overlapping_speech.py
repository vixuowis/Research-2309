# requirements_file --------------------

import subprocess

requirements = ["pyannote.audio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def detect_overlapping_speech(audio_file_path: str, access_token: str) -> List[Tuple[float, float]]:
    """
    Detect the segments of overlapping speech in an audio recording using the pyannote.audio framework.

    Args:
        audio_file_path (str): The path to the audio file that needs to be processed.
        access_token (str): The access token required to use the pre-trained model.

    Returns:
        List[Tuple[float, float]]: A list of tuples where each tuple contains the start and end time (in seconds) of each overlapping speech segment detected.

    Raises:
        RuntimeError: If the pre-trained model loading fails.
    """
    try:
        pipeline = Pipeline.from_pretrained('pyannote/overlapped-speech-detection', use_auth_token=access_token)
        output = pipeline(audio_file_path)
        overlapping_speech_segments = [(speech.start, speech.end) for speech in output.get_timeline().support()]
        return overlapping_speech_segments
    except Exception as e:
        raise RuntimeError(f"Failed to load the pre-trained model: {e}")

# test_function_code --------------------

def test_detect_overlapping_speech():
    print("Testing started.")
    # Assuming 'example.wav' is a valid audio file for testing and 'test_token' is a valid access token
    test_audio_file_path = 'example.wav'
    test_access_token = 'test_token'

    # Test case 1: Detect overlapping speech using a valid audio file and access token
    print("Testing case [1/1] started.")
    try:
        result = detect_overlapping_speech(test_audio_file_path, test_access_token)
        assert isinstance(result, list) and all(isinstance(segment, tuple) and len(segment) == 2 for segment in result), "Test case [1/1] failed: Result should be a list of tuples (start_time, end_time)."
    except RuntimeError as e:
        assert False, f"RuntimeError occurred: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_overlapping_speech()